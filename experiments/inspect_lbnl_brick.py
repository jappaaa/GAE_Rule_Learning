"""Throwaway inspection script for the LBNL Building 59 Brick model.

Not part of the pipeline -- just for exploring the TTL file's actual structure
(predicates, classes, sample relations) before designing the OfficeGraph-style
loader for this dataset. Run directly, or set breakpoints and run via the
"Python Debugger: Current File" launch config.
"""
import os
from collections import Counter

import rdflib

TTL_PATH = os.path.join('data', 'lbnl_b59', 'raw', 'Building_59', 'Bldg59_w_occ Brick model.ttl')

g = rdflib.Graph()
g.parse(TTL_PATH, format='turtle')
print(f"Total triples: {len(g)}")

print("\n--- Namespaces ---")
for prefix, ns in g.namespaces():
    print(f"  {prefix:10s} {ns}")

print("\n--- Predicate counts ---")
pred_counts = Counter(p for _, p, _ in g)
for pred, count in pred_counts.items():
    print(f"  {count:7d}  {pred}")

print("\n--- Class counts (rdf:type objects) ---")
type_counts = Counter(o for _, p, o in g if p == rdflib.RDF.type)
for cls, count in type_counts.items():
    print(f"  {count:7d}  {cls}")

# Pick one entity of each of a few classes and print its full neighborhood (both directions)
def show_entity(entity, label):
    print(f"\n--- Sample {label}: {entity} ---")
    print("  as subject:")
    for p, o in g.predicate_objects(entity):
        print(f"    {p}  ->  {o}")
    print("  as object:")
    for s, p in g.subject_predicates(entity):
        print(f"    {s}  ->  {p}")

# Grab one instance per class that looks zone/RTU/VAV/point-like, to inspect real relations
seen_labels = set()
for cls, _ in type_counts.most_common(15):
    cls_name = str(cls).rsplit('#', 1)[-1].rsplit('/', 1)[-1]
    if cls_name in seen_labels:
        continue
    seen_labels.add(cls_name)
    instances = list(g.subjects(rdflib.RDF.type, cls))
    if instances:
        show_entity(instances[0], cls_name)
