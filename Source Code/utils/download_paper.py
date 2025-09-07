"""
Script for splitting SDG queries, fetching Scopus results, and storing them in MongoDB.

USAGE & REQUIREMENTS:
---------------------
- You need a directory containing the full SDG query files (e.g., SDG01.txt, SDG02.txt, ...). Those are the Elsivier Scopus database SDG-related queries.
    (https://elsevier.digitalcommonsdata.com/datasets/y2zyy9vwzy/1)
- You must have a running MongoDB instance. A docker-compose file is provided in the 'database' directory of this project.
- You need a licensed Scopus API access. This is required to fetch data from Scopus.
- You must initialize pybliometrics with your Scopus API key(s):
    1. Run `pybliometrics init` in your terminal.
    2. Edit the generated .ini file (typically at ~/.pybliometrics/pybliometrics.cfg) and add your API key(s).
- Required Python packages (install with pip):
    pymongo
    pybliometrics
    anytree
    tqdm
    requests
    argparse

Example usage:
    python download_paper.py --sdg 1 --mongodb mongodb://localhost:27017 --shuffle

Arguments:
    --sdg        SDG number to process (e.g., 1 for SDG01.txt)
    --mongodb    MongoDB URI (default: mongodb://localhost:27017)
    --shuffle    Shuffle queries before processing (optional)

This script will:
- Parse and split the selected SDG query into sub-queries as mentioned in the paper appendix.
- Fetch results from Scopus for each sub-query.
- Store results in MongoDB, tracking which queries have been processed.
"""


import re
import time
import requests
import argparse
import os
import random
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError
from pybliometrics.scopus import ScopusSearch
from tqdm import tqdm
from anytree import Node, PreOrderIter

# ----------------------
# Query Parsing Utilities
# ----------------------
def split_query_into_minimal_units(query):
    """
    Splits a complex query into its minimal logical units based on parentheses and top-level ORs.
    """
    units = []
    current_unit = ''
    stack = []
    i = 0
    n = len(query)
    while i < n:
        char = query[i]
        if char == '(':  # Opening parenthesis
            stack.append('(')
            current_unit += char
        elif char == ')':  # Closing parenthesis
            if stack:
                stack.pop()
            current_unit += char
            if not stack:
                units.append(current_unit.strip())
                current_unit = ''
        elif query[i:i+2] == 'OR' and not stack:  # Top-level OR separator
            if current_unit.strip():
                units.append(current_unit.strip())
                current_unit = ''
            i += 1  # Skip 'O' and 'R'
        else:
            current_unit += char
        i += 1
    if current_unit.strip():
        units.append(current_unit.strip())
    return units

def tokenize(query):
    """
    Tokenizes a query string into logical and grouping tokens for parsing.
    """
    pattern = re.compile(r'''
        (AND\s+NOT)                     |            
        (\bAND\b|\bOR\b|\bNOT\b)        |           
        (\()                            |            
        (\))                            |            
        ([A-Z][A-Z0-9\-]*\s*                          
            \(\s*                                   
            (?:
                "(?:\\.|[^"\\])*"                  
                |
                [^()"']+                            
                |
                \(.*?\)                             
            )*?
        \))                                       
    ''', re.VERBOSE | re.IGNORECASE)
    tokens = []
    pos = 0
    while pos < len(query):
        match = pattern.match(query, pos)
        if match:
            groups = match.groups()
            token = next(g for g in groups if g)
            tokens.append(token.strip())
            pos = match.end()
        else:
            pos += 1
    return tokens

def parse_tree(tokens):
    """
    Parses a list of tokens into a tree structure representing the logical query.
    """
    root = Node("ROOT", type='group')
    current = root
    stack = []
    for token in tokens:
        token_upper = token.upper()
        if token_upper in {"AND", "OR", "NOT", "AND NOT"}:
            Node(token_upper, parent=current, type='logic')
        elif token == '(':  # Start new group
            new_node = Node("GROUP", parent=current, type='group')
            stack.append(current)
            current = new_node
        elif token == ')':  # End group
            if not stack:
                raise ValueError("Error! Invalid characters in the split. Check the main query text.")
            current = stack.pop()
        else:
            Node(token, parent=current, type='leaf')
    if stack:
        raise ValueError("Error! Invalid characters in the split. Check the main query text.")
    return root

def find_best_and_not_split(root, min_char_count=5000, min_node_count=10, max_depth=3):
    """
    Safely find a high-level AND NOT suitable for splitting.

    Constraints:
    - AND NOT must be within max_depth
    - AND NOT must separate meaningful sibling groups (not deep nesting)
    - Prefer the highest AND NOT with large subtree content

    """
    best_score = -1
    best_and_not = None
    best_negative = None
    for node in PreOrderIter(root):
        if node.name == "AND NOT" and node.type == "logic":
            if node.depth > max_depth:
                continue  # too nested, skip
            siblings = node.parent.children
            idx = siblings.index(node)
            if idx + 1 >= len(siblings):
                continue  # no right-hand negative subtree
            negative_root = siblings[idx + 1]
            positive_siblings = siblings[:idx]
            pos_char_count = sum(len(c.name) for s in positive_siblings for c in PreOrderIter(s))
            pos_node_count = sum(1 for s in positive_siblings for _ in PreOrderIter(s))
            neg_char_count = sum(len(c.name) for c in PreOrderIter(negative_root))
            neg_node_count = sum(1 for _ in PreOrderIter(negative_root))
            if (pos_char_count + neg_char_count < min_char_count and 
                pos_node_count + neg_node_count < min_node_count):
                continue  # too small, ignore
            score = (pos_char_count + neg_char_count) + (pos_node_count + neg_node_count) * 10
            if score > best_score:
                best_score = score
                best_and_not = node
                best_negative = negative_root
    return best_and_not, best_negative

def clone_subtree(node, new_parent):
    """
    Recursively clones a tree node and its children to a new parent node.
    """
    new_node = Node(node.name, type=node.type, parent=new_parent)
    for child in node.children:
        clone_subtree(child, new_node)

def collect_positive_subtree(root, and_not_node):
    """
    Collects the positive (non-AND-NOT) part of a query tree.
    """
    if and_not_node is None or and_not_node.parent is None:
        return root
    parent = and_not_node.parent
    idx = list(parent.children).index(and_not_node)
    new_root = Node("POSITIVE_ROOT", type='group')
    for child in parent.children[:idx]:
        clone_subtree(child, new_root)
    return new_root

def is_or_separated_group(node):
    """
    Checks if a group node is separated by OR operators.
    """
    children = node.children
    if len(children) < 3 or len(children) % 2 == 0:
        return False
    for i, child in enumerate(children):
        if i % 2 == 0 and child.type not in {"group", "leaf"}:
            return False
        if i % 2 == 1 and not (child.name == "OR" and child.type == "logic"):
            return False
    return True

def get_subtree_query_string(node):
    """
    Returns the concatenated string of all leaf nodes in a subtree.
    """
    return " ".join(n.name for n in PreOrderIter(node) if n.type == "leaf")

def recursive_or_splits(node, max_chars: int = 1500):
    """
    we split at the ORs level recursively, if the subgroup string is
    too long, more than 1500 characters, we go to see if it is split into ORs and split it in turn.
    """
    result_nodes = []
    def helper(current_node):
        if is_or_separated_group(current_node):
            total_str = get_subtree_query_string(current_node)
            if len(total_str) > max_chars:
                for i in range(0, len(current_node.children), 2):
                    child = current_node.children[i]
                    splits = recursive_or_splits(child, max_chars)
                    result_nodes.extend(splits)
                return  
        for child in current_node.children:
            helper(child)
    helper(node)
    return result_nodes if result_nodes else [node]

def smart_batch_positives_with_negatives(positives, negatives, max_positive_chars=1500, max_total_chars=2000):
    """
    Batches positive query strings and combines with negatives, ensuring length limits.
    """
    positive_batches = []
    current_batch = []
    current_length = 0
    for pos in positives:
        pos_len = len(pos) + (4 if current_batch else 0)  # +4 for " OR "
        if current_length + pos_len <= max_positive_chars:
            current_batch.append(pos)
            current_length += pos_len
        else:
            if current_batch:
                positive_batches.append(current_batch)
            current_batch = [pos]
            current_length = len(pos)
    if current_batch:
        positive_batches.append(current_batch)
    final_queries = []
    for batch in positive_batches:
        batch_str = " OR ".join(batch)
        if negatives:
            for neg in negatives:
                full_query = f"({batch_str}) AND NOT {neg}"
                if len(full_query) > max_total_chars:
                    for pos in batch:
                        mini_query = f"({pos}) AND NOT {neg}"
                        final_queries.append(mini_query)
                else:
                    final_queries.append(full_query)
        else:
            final_queries.append(batch_str)
    return final_queries

def sanitize_query_string(q):
    """
    Cleans up the query string for Scopus API compatibility.
    """
    return re.sub(r'([A-Z][A-Z0-9\-]*)\s+\(', r'\1(', q)

def split_main(query: str):    
    """
    Splits a main SDG query into smaller sub-queries for efficient processing.

    This method takes a Scopus SDG query and generates a split of it.
    The system creates trees for each macro component and then splits the query so that each sub-query
    does not exceed approximately 2000 characters. To maintain the robustness of the approach, we exploit the commutative
    and associative of the OR operators. The final list of queries allows us to reformulate exactly the result of the integer query
    by going to create the union of the sets. 

    """
    main_or_splits = split_query_into_minimal_units(query)
    total_queries = []
    for i, splits in enumerate(main_or_splits):
        try:
            tokens = tokenize(splits)
            main_root = parse_tree(tokens)
            possible_negative_node, negative_root = find_best_and_not_split(main_root)
            positive_root = collect_positive_subtree(main_root, possible_negative_node)
            positive_elements = recursive_or_splits(positive_root)
            positives = [stringify_node(c) for c in positive_elements]
            if negative_root:
                negative_elements = recursive_or_splits(negative_root)
                negatives = [stringify_node(c) for c in negative_elements]
            else:
                negative_elements = None
                negatives = None
            queries = smart_batch_positives_with_negatives(positives, negatives)
            total_queries.extend([sanitize_query_string(c) for c in queries])
        except Exception as e:
            print(f"Error at {i} split. Error: {e}. Skip.")
    return total_queries

def format_pybliometrics_result(result):
    """
    Formats a pybliometrics ScopusSearch result into a dictionary for MongoDB.
    """
    return {
        "eid": result.eid,
        "title": result.title,
        "publicationName": result.publicationName,
        "coverDate": result.coverDate,
        "doi": result.doi,
        "description": result.description,
        "citation_count": result.citedby_count
    }

def get_cursor_based_results(sub_query):
    """
    Uses pybliometrics to fetch Scopus results for a sub-query.
    """
    search = ScopusSearch(sub_query, refresh=True, date="2015-2022")
    results = search.results or []
    return [format_pybliometrics_result(r) for r in results]

def insert_documents_bulk(docs, sdg_label, query_idx=None):
    """
    Inserts or updates documents in MongoDB in bulk, tagging with the SDG label.
    """
    if not docs:
        print(f"Query {query_idx}: No results.\n" if query_idx else "No results.\n")
        return
    eids = [doc["eid"] for doc in docs]
    existing_docs = collection.find({"eid": {"$in": eids}}, {"eid": 1, "sdgs": 1})
    existing_eid_to_sdgs = {doc["eid"]: doc.get("sdgs", []) for doc in existing_docs}
    operations = []
    inserted, updated, skipped = 0, 0, 0
    for doc in docs:
        eid = doc["eid"]
        current_sdgs = existing_eid_to_sdgs.get(eid)
        if current_sdgs is None:
            doc["sdgs"] = [sdg_label]
            operations.append(UpdateOne(
                {"eid": eid},
                {"$setOnInsert": doc},
                upsert=True
            ))
            inserted += 1
        elif sdg_label not in current_sdgs:
            operations.append(UpdateOne(
                {"eid": eid},
                {"$addToSet": {"sdgs": sdg_label}}
            ))
            updated += 1
        else:
            skipped += 1
    if operations:
        try:
            collection.bulk_write(operations, ordered=False)
        except BulkWriteError as e:
            print(f"Bulk write error in query {query_idx}: {e.details}")
    print(f"Query {query_idx}: Inserted {inserted}, Updated {updated}, Skipped {skipped}\n" if query_idx else f"Inserted {inserted}, Updated {updated}, Skipped {skipped}\n")

def smart_scopus_search_and_store(sub_query, sdg_label, query_idx=None):
    """
    Fetches Scopus results for a sub-query and stores them in MongoDB.
    """
    try:
        print(f"Query {query_idx}, sdg_label: {sdg_label}")
        docs = get_cursor_based_results(sub_query)
        insert_documents_bulk(docs, sdg_label, query_idx)
    except Exception as e:
        print(f"Query {query_idx} failed. Error: {e}")

def mark_as_processed(sdg_label, query):
    """
    Marks a query as processed in the tracking collection.
    """
    tracking_collection.insert_one({
        "sdg_label": sdg_label,
        "query": query,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })

# ----------------------
# Main Execution Block
# ----------------------
if __name__ == "__main__":
    import pybliometrics
    pybliometrics.init()
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Download and process SDG papers.")
    parser.add_argument('--sdg', type=int, required=True, help='SDG number to process (e.g., 1 for SDG01.txt)')
    parser.add_argument('--mongodb', type=str, default='mongodb://localhost:27017', help='MongoDB URI')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle queries before processing')
    args = parser.parse_args()
    # Connect to MongoDB
    client = MongoClient(args.mongodb)
    db = client['sdg_paper']
    collection = db['sdgg']
    tracking_collection = db["processed_queries"]
    # Prepare SDG query file path
    sdg_label = args.sdg
    sdg_filename = f"SDG{sdg_label:02d}.txt"
    sdg_path = os.path.join(os.path.dirname(__file__), '../../Supporting Materials and Tools/SDG_queries', sdg_filename)
    sdg_path = os.path.abspath(sdg_path)
    if not os.path.exists(sdg_path):
        raise FileNotFoundError(f"SDG query file not found: {sdg_path}")
    # Read and split the main SDG query
    with open(sdg_path, "r") as f:
        query = f.read()
    queries = split_main(query)
    if args.shuffle:
        random.shuffle(queries)
    print(f"There are {len(queries)} queries for SDG {sdg_label}")
    # Process each sub-query
    for idx, query in enumerate(queries, start=1):
        smart_scopus_search_and_store(query, sdg_label, idx)
        mark_as_processed(sdg_label, query)
