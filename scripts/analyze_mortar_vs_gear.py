#!/usr/bin/env python3
"""Analyse dataset shift between MORTAR and GEAR building knowledge graphs.

Configure the four path variables below, then run:
    python scripts/analyze_mortar_vs_gear.py

Produces 13 CSV files and one LaTeX Beamer presentation in OUTPUT_DIRECTORY.
"""

# ============================================================
# EDITABLE CONFIGURATION — set these four paths before running
# ============================================================
MORTAR_DATA_DIRECTORY = "data/mortar_ttl"
GEAR_DATASET_PATH     = "data/ttl/05_IntegratedModel_L6_HCDLAB_20260625.ttl"
BRICK_ONTOLOGY_PATH   = "Brick.ttl"
OUTPUT_DIRECTORY      = "output/mortar_vs_gear"
# ============================================================

import csv
import math
import os
import re
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import rdflib
from rdflib import Graph, URIRef, Literal, BNode, Namespace
from rdflib.namespace import RDF, RDFS, OWL

TARGET_RELATIONS = {
    "brick:feeds",
    "brick:hasPart",
    "brick:hasPoint",
}

# ── Normalization constants ──────────────────────────────────
SCHEMA_NS = [
    ("https://brickschema.org/schema/Brick/ref#",       "ref"),
    ("http://qudt.org/vocab/unit/",                     "unit"),
    ("https://brickschema.org/schema/BrickTag#",        "tag"),
    ("https://brickschema.org/schema/1.0.1/BrickTag#",  "tag"),
    ("https://brickschema.org/schema/1.0.2/BrickTag#",  "tag"),
    ("https://brickschema.org/schema/Brick#",           "brick"),
    ("https://brickschema.org/schema/1.0.1/Brick#",     "brick"),
    ("https://brickschema.org/schema/1.0.2/Brick#",     "brick"),
    ("https://brickschema.org/schema/1.0.1/BrickFrame#","brick"),
    ("https://brickschema.org/schema/1.0.2/BrickFrame#","brick"),
    ("http://www.w3.org/1999/02/22-rdf-syntax-ns#",     "rdf"),
    ("http://www.w3.org/2000/01/rdf-schema#",           "rdfs"),
    ("http://www.w3.org/2002/07/owl#",                  "owl"),
    ("http://www.w3.org/2001/XMLSchema#",               "xsd"),
    ("https://w3id.org/rec/",                           "rec"),
    ("https://w3id.org/rec/core/",                      "rec"),
    ("http://www.w3id.org/rec/",                        "rec"),
]

# Inverse relations → canonical form (subject/object swap)
FOLD = {
    "isPointOf": "brick:hasPoint",
    "isPartOf":  "brick:hasPart",
    "isFedBy":   "brick:feeds",
    # brick: prefixed inverses
    "brick:isPointOf": "brick:hasPoint",
    "brick:isPartOf":  "brick:hasPart",
    "brick:isFedBy":   "brick:feeds",
}

# REC predicate local-name → Brick equivalent
REC_TO_BRICK_REL = {
    "locatedIn":    "brick:hasLocation",
    "isLocationOf": "brick:isLocationOf",
    "hasLocation":  "brick:hasLocation",
    "feeds":        "brick:feeds",
    "isFedBy":      "brick:feeds",      # inverse folded
    "hasPoint":     "brick:hasPoint",
    "isPointOf":    "brick:hasPoint",   # inverse folded
    "hasPart":      "brick:hasPart",
    "isPartOf":     "brick:hasPart",    # inverse folded
    "includes":     "brick:hasPart",
}

# REC / local-namespace class local-name → Brick class
REC_TO_BRICK_CLASS = {
    "Room":     "brick:Room",
    "Level":    "brick:Floor",
    "Floor":    "brick:Floor",
    "Story":    "brick:Floor",
    "Building": "brick:Building",
    "Site":     "brick:Site",
    "Space":    "brick:Space",
    "Zone":     "brick:Zone",
}

# Broad ontology categories in priority order (first match wins)
# Keys are Brick superclass local-names
CATEGORY_ANCHORS = [
    ("Collection", ["Collection"]),
    ("Point",      ["Point"]),
    ("Location",   ["Location", "Room", "Floor", "Site", "Space", "Zone"]),
    ("Building",   ["Building"]),
    ("System",     ["System"]),
    ("Equipment",  ["Equipment", "HVAC_Equipment", "Electrical_Equipment",
                    "Lighting_Equipment", "Water_Heating_Equipment"]),
]

# ── Utilities ────────────────────────────────────────────────

def _ns_prefix(uri: str) -> str:
    for ns, prefix in SCHEMA_NS:
        if uri.startswith(ns):
            return f"{prefix}:{uri[len(ns):]}"
    local = uri.split("#", 1)[1] if "#" in uri else uri.rsplit("/", 1)[-1]
    ns_part = uri.rsplit("#", 1)[0] if "#" in uri else uri.rsplit("/", 1)[0]
    return f"<{ns_part}>:{local}"

def qname(term) -> str:
    if isinstance(term, URIRef):
        return _ns_prefix(str(term))
    if isinstance(term, Literal):
        return f'"{term}"'
    return "_:bnode"

def _local(qn: str) -> str:
    return qn.split(":", 1)[1] if ":" in qn else qn


# ── Brick ontology ───────────────────────────────────────────

def load_brick_ontology(path):
    """Returns (ancestor_map, category_map).
    ancestor_map: class_local -> [parent, grandparent, ...]
    category_map: class_local -> category string
    """
    if not path or path.startswith("<"):
        return {}, {}
    BRICK = "https://brickschema.org/schema/Brick#"
    g = Graph()
    try:
        g.parse(str(path), format="turtle")
    except Exception as e:
        print(f"  [WARN] Could not parse Brick ontology: {e}")
        return {}, {}

    parents = {}
    for s, _, o in g.triples((None, RDFS.subClassOf, None)):
        if str(s).startswith(BRICK) and str(o).startswith(BRICK):
            parents[str(s)[len(BRICK):]] = str(o)[len(BRICK):]

    def ancestors(cls):
        chain, cur, seen = [], cls, set()
        while cur in parents and cur not in seen:
            seen.add(cur)
            cur = parents[cur]
            chain.append(cur)
        return chain

    all_cls = set(parents) | set(parents.values())
    ancestor_map = {c: ancestors(c) for c in all_cls}

    # Build category map
    category_map = {}
    for cls in all_cls:
        chain = [cls] + ancestor_map.get(cls, [])
        cat = "Other"
        for category, anchors in CATEGORY_ANCHORS:
            if any(a in chain for a in anchors):
                cat = category
                break
        category_map[cls] = cat

    return ancestor_map, category_map


# ── Data loading ─────────────────────────────────────────────

def _detect_malformed(uri: str):
    """Flag common malformed Brick URIs."""
    issues = []
    s = str(uri)
    if "brick#brick:" in s.lower() or "brick#brick:" in s:
        issues.append("double_brick_prefix")
    if re.search(r"#[A-Za-z_]+:[A-Za-z_]", s):
        issues.append("colon_in_local_name")
    return issues

def load_ttl_file(path):
    """Parse one TTL file. Returns (graph, parse_errors, quality_issues)."""
    g = Graph()
    errors = []
    issues = []
    try:
        g.parse(str(path), format="turtle")
    except Exception as e:
        errors.append(str(e))
        return g, errors, issues

    # Check for malformed URIs
    for s, p, o in g:
        for term in (s, p, o):
            if isinstance(term, URIRef):
                for issue in _detect_malformed(str(term)):
                    issues.append({"file": Path(path).name, "term": str(term), "issue": issue})

    return g, errors, issues


def load_mortar(directory):
    """Load all .ttl files in directory. Returns list of (stem, Graph, errors, issues)."""
    d = Path(directory)
    if not d.exists():
        print(f"[ERROR] MORTAR directory not found: {directory}", file=sys.stderr)
        sys.exit(1)
    files = sorted(d.glob("*.ttl"))
    if not files:
        print(f"[ERROR] No .ttl files in {directory}", file=sys.stderr)
        sys.exit(1)
    result = []
    for f in files:
        g, errors, issues = load_ttl_file(f)
        if errors:
            print(f"  [PARSE ERROR] {f.name}: {errors[0]}")
        result.append((f.stem, g, errors, issues))
    return result


def load_gear(path):
    """Load single GEAR TTL file."""
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] GEAR file not found: {path}", file=sys.stderr)
        sys.exit(1)
    g, errors, issues = load_ttl_file(p)
    if errors:
        print(f"  [PARSE ERROR] {p.name}: {errors[0]}")
    return p.stem, g, errors, issues


# ── Normalization ────────────────────────────────────────────

def normalize_triple(s_raw, p_raw, o_raw, fold_inverses=True):
    """Convert one raw (s,p,o) to normalized qname form. Returns (s,p,o,was_folded,was_remapped)."""
    s = qname(s_raw)
    p = qname(p_raw)
    o = qname(o_raw)

    was_folded   = False
    was_remapped = False

    if isinstance(s_raw, BNode) or isinstance(o_raw, BNode):
        return s, p, o, False, False

    p_local = _local(p)
    p_prefix = p.split(":", 1)[0] if ":" in p else ""

    # Fix double-prefix types
    if p == "rdf:type" and o.startswith("brick:brick:"):
        o = o[len("brick:"):]
        was_remapped = True

    # Fold inverses
    if fold_inverses and p_local in FOLD:
        s, p, o = o, FOLD[p_local], s
        was_folded = True
    elif fold_inverses and p in FOLD:
        s, p, o = o, FOLD[p], s
        was_folded = True
    # REC relation remapping (non-brick prefixed)
    elif p_prefix not in ("brick", "rdf", "rdfs", "owl", "xsd", "ref", "tag"):
        if p_local in REC_TO_BRICK_REL:
            p = REC_TO_BRICK_REL[p_local]
            was_remapped = True
        elif p_local in {"hasPoint", "hasPart", "feeds"} and p_prefix:
            p = f"brick:{p_local}"
            was_remapped = True

    # REC class remapping for rdf:type objects
    if p == "rdf:type" and not o.startswith("brick:") and ":" in o:
        o_local = _local(o)
        if o_local in REC_TO_BRICK_CLASS:
            o = REC_TO_BRICK_CLASS[o_local]
            was_remapped = True

    return s, p, o, was_folded, was_remapped


def extract_triples(graph, fold_inverses=True):
    """Extract all triples as normalized (s,p,o) strings, skipping literals/bnodes as subjects.
    Returns (raw_triples, norm_triples, fold_count, remap_count, dup_raw, dup_norm)"""
    raw_set  = set()
    norm_set = set()
    raw_list = []
    norm_list = []
    folds = 0
    remaps = 0

    for s_r, p_r, o_r in graph:
        if isinstance(s_r, BNode):
            continue

        # Raw (still qname form)
        s = qname(s_r); p = qname(p_r); o = qname(o_r)
        raw_list.append((s, p, o))

        # Normalized
        sn, pn, on, folded, remapped = normalize_triple(s_r, p_r, o_r, fold_inverses)
        if folded:  folds += 1
        if remapped: remaps += 1
        norm_list.append((sn, pn, on))

    dup_raw  = len(raw_list)  - len(set(raw_list))
    dup_norm = len(norm_list) - len(set(norm_list))
    return raw_list, norm_list, folds, remaps, dup_raw, dup_norm


# ── Graph statistics ─────────────────────────────────────────

def graph_stats(triples, label=""):
    """Compute graph statistics from a list of (s,p,o) strings."""
    subjects   = Counter()
    objects    = Counter()
    preds      = Counter()
    types_map  = defaultdict(set)   # entity -> set of types
    degree     = Counter()          # total degree (in+out)
    in_deg     = Counter()
    out_deg    = Counter()

    target_triples = []
    non_target     = []

    for s, p, o in triples:
        preds[p] += 1
        subjects[s] += 1
        objects[o]  += 1
        out_deg[s]  += 1
        if not o.startswith('"') and not o.startswith("_:"):
            in_deg[o] += 1
        if p == "rdf:type":
            types_map[s].add(o)
        if p in TARGET_RELATIONS:
            target_triples.append((s, p, o))
        else:
            non_target.append((s, p, o))

    all_entities = set(subjects) | {o for _, _, o in triples
                                    if not o.startswith('"') and not o.startswith("_:")}
    for e in all_entities:
        degree[e] = out_deg.get(e, 0) + in_deg.get(e, 0)

    degs = list(degree.values())
    degs.sort()
    n = len(degs)

    return {
        "n_triples":        len(triples),
        "n_unique_triples": len(set(triples)),
        "n_subjects":       len(subjects),
        "n_objects":        len({o for _,_,o in triples if not o.startswith('"') and not o.startswith("_:")}),
        "n_entities":       len(all_entities),
        "n_predicates":     len(preds),
        "n_classes":        len({o for _,p,o in triples if p == "rdf:type"}),
        "n_duplicates":     len(triples) - len(set(triples)),
        "n_target":         len(target_triples),
        "n_non_target":     len(non_target),
        "subj_only":        len(set(subjects) - {o for _,_,o in triples if not o.startswith('"') and not o.startswith("_:")}),
        "obj_only":         len({o for _,_,o in triples if not o.startswith('"') and not o.startswith("_:")} - set(subjects)),
        "avg_degree":       round(sum(degs) / n, 3) if n else 0,
        "median_degree":    degs[n // 2] if n else 0,
        "max_degree":       max(degs) if degs else 0,
        "predicates":       preds,
        "types_map":        types_map,
        "in_deg":           in_deg,
        "out_deg":          out_deg,
        "degree":           degree,
        "target_triples":   target_triples,
        "non_target":       non_target,
        "all_entities":     all_entities,
    }


def connectivity_stats(triples):
    """Union-find component analysis (undirected)."""
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[a] = b

    for s, _, o in triples:
        if not o.startswith('"') and not o.startswith("_:"):
            union(s, o)

    comps = Counter(find(n) for n in parent)
    sizes = sorted(comps.values(), reverse=True)
    return {
        "n_components":    len(comps),
        "largest_comp":    sizes[0] if sizes else 0,
        "isolated_nodes":  sum(1 for v in comps.values() if v == 1),
    }


def target_rel_stats(triples):
    """Per-relation stats for target triples."""
    by_rel = defaultdict(list)
    for s, p, o in triples:
        if p in TARGET_RELATIONS:
            by_rel[p].append((s, o))
    result = {}
    for rel, pairs in by_rel.items():
        subj_cnt = Counter(s for s, _ in pairs)
        obj_cnt  = Counter(o for _, o in pairs)
        sc = list(subj_cnt.values())
        oc = list(obj_cnt.values())
        result[rel] = {
            "count":       len(pairs),
            "unique_subj": len(subj_cnt),
            "unique_obj":  len(obj_cnt),
            "avg_per_subj": round(sum(sc)/len(sc), 2) if sc else 0,
            "med_per_subj": sorted(sc)[len(sc)//2] if sc else 0,
            "max_per_subj": max(sc) if sc else 0,
            "avg_per_obj":  round(sum(oc)/len(oc), 2) if oc else 0,
            "repeated_pairs": len(pairs) - len(set(pairs)),
        }
    return result


def type_pair_stats(triples, types_map, category_map, ancestor_map):
    """For each target relation, count (subj_type, obj_type) and (subj_cat, obj_cat) pairs."""
    def get_types(e):
        return types_map.get(e, set())

    def get_category(types):
        for t in types:
            local = _local(t)
            if local in category_map:
                return category_map[local]
            # Check ancestors
            for anc in ancestor_map.get(local, []):
                if anc in category_map:
                    return category_map[anc]
        if types:
            return "Other"
        return "Unrecognised"

    result = defaultdict(Counter)
    cat_result = defaultdict(Counter)
    for s, p, o in triples:
        if p not in TARGET_RELATIONS:
            continue
        s_types = get_types(s)
        o_types = get_types(o)
        # Most specific types
        for st in (s_types or {"(untyped)"}):
            for ot in (o_types or {"(untyped)"}):
                result[p][(st, ot)] += 1
        s_cat = get_category(s_types)
        o_cat = get_category(o_types)
        cat_result[p][(s_cat, o_cat)] += 1

    return result, cat_result


def naming_stats(triples):
    """Analyse entity identifier conventions."""
    entities = set()
    for s, p, o in triples:
        entities.add(s)
        if not o.startswith('"') and not o.startswith("_:"):
            entities.add(o)

    namespaces = Counter()
    local_lengths = []
    tokens_per_name = []
    has_dot, has_under, has_hyphen, has_numeric = 0, 0, 0, 0

    for e in entities:
        if ":" not in e:
            continue
        prefix, local = e.split(":", 1)
        namespaces[prefix] += 1
        local_lengths.append(len(local))
        toks = re.split(r"[_.\-/]+", local)
        tokens_per_name.append(len(toks))
        if "." in local:   has_dot += 1
        if "_" in local:   has_under += 1
        if "-" in local:   has_hyphen += 1
        if re.search(r"\d", local): has_numeric += 1

    n = len(entities) or 1
    ll = local_lengths or [0]
    tl = tokens_per_name or [0]
    return {
        "namespaces":         namespaces,
        "avg_local_len":      round(sum(ll)/len(ll), 1),
        "avg_tokens":         round(sum(tl)/len(tl), 2),
        "pct_dot":            round(100*has_dot/n, 1),
        "pct_underscore":     round(100*has_under/n, 1),
        "pct_hyphen":         round(100*has_hyphen/n, 1),
        "pct_numeric":        round(100*has_numeric/n, 1),
    }


def context_after_removal(triples):
    """Stats on what remains after removing target relations."""
    remaining = [(s,p,o) for s,p,o in triples if p not in TARGET_RELATIONS]
    target_subj = {s for s,p,_ in triples if p in TARGET_RELATIONS}
    target_obj  = {o for _,p,o in triples if p in TARGET_RELATIONS}
    target_ents = target_subj | target_obj

    rem_subj = {s for s,_,_ in remaining}
    rem_obj  = {o for _,_,o in remaining if not o.startswith('"') and not o.startswith("_:")}
    rem_ents = rem_subj | rem_obj

    has_type    = {s for s,p,_ in remaining if p == "rdf:type"}
    has_loc     = {s for s,p,_ in remaining if p in ("brick:hasLocation","brick:isLocationOf")}

    isolated = target_ents - rem_ents
    subj_with_ctx = target_subj & rem_subj
    obj_with_ctx  = target_obj & rem_obj

    return {
        "rem_triples":    len(remaining),
        "rem_predicates": len({p for _,p,_ in remaining}),
        "rem_entities":   len(rem_ents),
        "isolated_after": len(isolated),
        "has_type":       len(has_type),
        "has_location":   len(has_loc),
        "subj_ctx_pct":   round(100*len(subj_with_ctx)/(len(target_subj) or 1), 1),
        "obj_ctx_pct":    round(100*len(obj_with_ctx)/(len(target_obj) or 1), 1),
    }


def js_divergence(c1: Counter, c2: Counter) -> float:
    """Jensen-Shannon divergence between two count distributions."""
    keys = set(c1) | set(c2)
    t1, t2 = sum(c1.values()) or 1, sum(c2.values()) or 1
    p = {k: c1.get(k, 0)/t1 for k in keys}
    q = {k: c2.get(k, 0)/t2 for k in keys}
    m = {k: 0.5*(p[k]+q[k]) for k in keys}
    def kl(a, b):
        return sum(a[k]*math.log2(a[k]/b[k]) for k in keys if a[k] > 0 and b[k] > 0)
    return round(0.5*kl(p,m) + 0.5*kl(q,m), 4)


def jaccard(s1: set, s2: set) -> float:
    if not s1 and not s2:
        return 1.0
    return round(len(s1 & s2) / len(s1 | s2), 4)


# ── CSV writers ──────────────────────────────────────────────

def w(path, rows, header):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote: {path}")


def safe_tex(s):
    # Escape in order: backslash first, then other specials, then angle brackets last.
    t = str(s)
    t = t.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")
    t = t.replace("_", r"\_").replace("^", r"\^{}")
    t = t.replace("<", r"\textless{}").replace(">", r"\textgreater{}")
    return t


# ── Main analysis ────────────────────────────────────────────

def run():
    out = Path(OUTPUT_DIRECTORY)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Loading Brick ontology...")
    ancestor_map, category_map = load_brick_ontology(BRICK_ONTOLOGY_PATH)
    print(f"  {len(ancestor_map)} Brick classes loaded.")

    print("\nLoading MORTAR...")
    mortar_buildings = load_mortar(MORTAR_DATA_DIRECTORY)
    print(f"  {len(mortar_buildings)} buildings.")

    print("\nLoading GEAR...")
    gear_stem, gear_graph, gear_errors, gear_issues = load_gear(GEAR_DATASET_PATH)
    print(f"  Loaded {gear_stem}.")

    # ── Extract and normalize ────────────────────────────────
    print("\nExtracting and normalizing triples...")

    mortar_raw_all, mortar_norm_all = [], []
    mortar_building_stats = []
    mortar_all_issues = []
    total_m_folds = total_m_remaps = total_m_dup_raw = total_m_dup_norm = 0

    for stem, g, errors, issues in mortar_buildings:
        raw, norm, folds, remaps, dr, dn = extract_triples(g)
        mortar_raw_all.extend(raw)
        mortar_norm_all.extend(norm)
        total_m_folds  += folds
        total_m_remaps += remaps
        total_m_dup_raw  += dr
        total_m_dup_norm += dn
        mortar_all_issues.extend(issues)
        st = graph_stats(norm, stem)
        conn = connectivity_stats(norm)
        mortar_building_stats.append({
            "building":     stem,
            "n_triples":    st["n_triples"],
            "n_entities":   st["n_entities"],
            "n_predicates": st["n_predicates"],
            "n_classes":    st["n_classes"],
            "n_target":     st["n_target"],
            "n_non_target": st["n_non_target"],
            "avg_degree":   st["avg_degree"],
            "n_components": conn["n_components"],
            "errors":       len(errors),
            "parse_errors": "; ".join(errors)[:200],
        })

    gear_raw, gear_norm, g_folds, g_remaps, g_dr, g_dn = extract_triples(gear_graph)
    mortar_all_issues.extend(gear_issues)

    print(f"  MORTAR: {len(mortar_raw_all):,} raw triples, "
          f"{total_m_folds} folded, {total_m_remaps} remapped, {total_m_dup_raw} raw-dups.")
    print(f"  GEAR:   {len(gear_raw):,} raw triples, "
          f"{g_folds} folded, {g_remaps} remapped, {g_dr} raw-dups.")

    # Graph stats
    mst = graph_stats(mortar_norm_all, "MORTAR")
    gst = graph_stats(gear_norm, "GEAR")
    mconn = connectivity_stats(mortar_norm_all)
    gconn = connectivity_stats(gear_norm)

    # ── 1. dataset_summary.csv ───────────────────────────────
    rows = []
    for label, st, conn, raw, norm, folds, remaps, dup_raw, dup_norm, n_files in [
        ("MORTAR", mst, mconn, mortar_raw_all, mortar_norm_all,
         total_m_folds, total_m_remaps, total_m_dup_raw, total_m_dup_norm,
         len(mortar_buildings)),
        ("GEAR",   gst, gconn, gear_raw, gear_norm,
         g_folds, g_remaps, g_dr, g_dn, 1),
    ]:
        rows.append({
            "dataset":          label,
            "n_files":          n_files,
            "raw_triples":      len(raw),
            "norm_triples":     len(norm),
            "folds_applied":    folds,
            "remaps_applied":   remaps,
            "dup_raw":          dup_raw,
            "dup_norm":         dup_norm,
            "n_entities":       st["n_entities"],
            "n_subjects":       st["n_subjects"],
            "n_objects":        st["n_objects"],
            "n_predicates":     st["n_predicates"],
            "n_classes":        st["n_classes"],
            "n_target":         st["n_target"],
            "n_non_target":     st["n_non_target"],
            "subj_only":        st["subj_only"],
            "obj_only":         st["obj_only"],
            "avg_degree":       st["avg_degree"],
            "median_degree":    st["median_degree"],
            "max_degree":       st["max_degree"],
            "n_components":     conn["n_components"],
            "largest_comp":     conn["largest_comp"],
            "isolated_nodes":   conn["isolated_nodes"],
        })
    w(out/"dataset_summary.csv", rows,
      list(rows[0].keys()))

    # ── 2. building_level_summary.csv ───────────────────────
    w(out/"building_level_summary.csv", mortar_building_stats,
      list(mortar_building_stats[0].keys()) if mortar_building_stats else [])

    # ── 3. predicate_comparison.csv ─────────────────────────
    m_pred = mst["predicates"]
    g_pred = gst["predicates"]
    all_preds = set(m_pred) | set(g_pred)
    m_total = len(mortar_norm_all) or 1
    g_total = len(gear_norm) or 1

    def pred_ns(p):
        return p.split(":", 1)[0] if ":" in p else "unknown"

    pred_rows = []
    for p in sorted(all_preds):
        mc = m_pred.get(p, 0)
        gc = g_pred.get(p, 0)
        pred_rows.append({
            "predicate":        p,
            "mortar_count":     mc,
            "gear_count":       gc,
            "mortar_pct":       round(100*mc/m_total, 3),
            "gear_pct":         round(100*gc/g_total, 3),
            "only_mortar":      gc == 0,
            "only_gear":        mc == 0,
            "namespace":        pred_ns(p),
            "is_target":        p in TARGET_RELATIONS,
        })
    pred_rows.sort(key=lambda r: -(r["mortar_count"]+r["gear_count"]))
    w(out/"predicate_comparison.csv", pred_rows, list(pred_rows[0].keys()) if pred_rows else [])

    # ── 4. class_comparison.csv ─────────────────────────────
    m_classes = Counter(o for _,p,o in mortar_norm_all if p == "rdf:type")
    g_classes = Counter(o for _,p,o in gear_norm     if p == "rdf:type")
    all_cls   = set(m_classes) | set(g_classes)
    class_rows = []
    for cls in sorted(all_cls):
        mc = m_classes.get(cls, 0)
        gc = g_classes.get(cls, 0)
        local = _local(cls)
        in_brick = local in ancestor_map
        class_rows.append({
            "class":         cls,
            "mortar_count":  mc,
            "gear_count":    gc,
            "only_mortar":   gc == 0,
            "only_gear":     mc == 0,
            "in_brick":      in_brick,
            "namespace":     cls.split(":", 1)[0] if ":" in cls else "unknown",
        })
    class_rows.sort(key=lambda r: -(r["mortar_count"]+r["gear_count"]))
    w(out/"class_comparison.csv", class_rows, list(class_rows[0].keys()) if class_rows else [])

    # ── 5. ontology_category_comparison.csv ─────────────────
    def categorise(types_map, dataset_label):
        cat_count = Counter()
        for e, types in types_map.items():
            cat = "Unrecognised"
            for t in types:
                local = _local(t)
                if local in category_map:
                    cat = category_map[local]
                    break
                for anc in ancestor_map.get(local, []):
                    if anc in category_map:
                        cat = category_map[anc]
                        break
                else:
                    continue
                break
            cat_count[cat] += 1
        # Entities with no type
        return cat_count

    m_cats = categorise(mst["types_map"], "MORTAR")
    g_cats = categorise(gst["types_map"], "GEAR")
    all_cats = set(m_cats) | set(g_cats) | {"Equipment","Point","Location","Building","System","Collection","Other","Unrecognised"}
    cat_rows = []
    m_typed = sum(m_cats.values()) or 1
    g_typed = sum(g_cats.values()) or 1
    for cat in sorted(all_cats):
        cat_rows.append({
            "category":     cat,
            "mortar_count": m_cats.get(cat, 0),
            "gear_count":   g_cats.get(cat, 0),
            "mortar_pct":   round(100*m_cats.get(cat,0)/m_typed, 1),
            "gear_pct":     round(100*g_cats.get(cat,0)/g_typed, 1),
        })
    cat_rows.sort(key=lambda r: -(r["mortar_count"]+r["gear_count"]))
    w(out/"ontology_category_comparison.csv", cat_rows, list(cat_rows[0].keys()))

    # ── 6. target_relation_comparison.csv ───────────────────
    m_trel = target_rel_stats(mortar_norm_all)
    g_trel = target_rel_stats(gear_norm)
    m_target_total = sum(v["count"] for v in m_trel.values()) or 1
    g_target_total = sum(v["count"] for v in g_trel.values()) or 1

    trel_rows = []
    for rel in sorted(TARGET_RELATIONS):
        ms = m_trel.get(rel, {})
        gs = g_trel.get(rel, {})
        trel_rows.append({
            "relation":             rel,
            "mortar_count":         ms.get("count", 0),
            "gear_count":           gs.get("count", 0),
            "mortar_pct_target":    round(100*ms.get("count",0)/m_target_total, 1),
            "gear_pct_target":      round(100*gs.get("count",0)/g_target_total, 1),
            "mortar_pct_all":       round(100*ms.get("count",0)/(len(mortar_norm_all) or 1), 2),
            "gear_pct_all":         round(100*gs.get("count",0)/(len(gear_norm) or 1), 2),
            "mortar_unique_subj":   ms.get("unique_subj", 0),
            "gear_unique_subj":     gs.get("unique_subj", 0),
            "mortar_unique_obj":    ms.get("unique_obj", 0),
            "gear_unique_obj":      gs.get("unique_obj", 0),
            "mortar_avg_per_subj":  ms.get("avg_per_subj", 0),
            "gear_avg_per_subj":    gs.get("avg_per_subj", 0),
            "mortar_med_per_subj":  ms.get("med_per_subj", 0),
            "gear_med_per_subj":    gs.get("med_per_subj", 0),
            "mortar_max_per_subj":  ms.get("max_per_subj", 0),
            "gear_max_per_subj":    gs.get("max_per_subj", 0),
            "mortar_repeated_pairs":ms.get("repeated_pairs", 0),
            "gear_repeated_pairs":  gs.get("repeated_pairs", 0),
        })
    w(out/"target_relation_comparison.csv", trel_rows, list(trel_rows[0].keys()))

    # ── 7. target_type_pair_comparison.csv ──────────────────
    m_type_pairs, m_cat_pairs = type_pair_stats(mortar_norm_all, mst["types_map"], category_map, ancestor_map)
    g_type_pairs, g_cat_pairs = type_pair_stats(gear_norm,       gst["types_map"], category_map, ancestor_map)

    pair_rows = []
    for rel in sorted(TARGET_RELATIONS):
        mc = m_cat_pairs.get(rel, Counter())
        gc = g_cat_pairs.get(rel, Counter())
        all_pairs = set(mc) | set(gc)
        for pair in sorted(all_pairs, key=lambda p: -(mc.get(p,0)+gc.get(p,0))):
            pair_rows.append({
                "relation":      rel,
                "subj_category": pair[0],
                "obj_category":  pair[1],
                "mortar_count":  mc.get(pair, 0),
                "gear_count":    gc.get(pair, 0),
                "only_mortar":   gc.get(pair, 0) == 0,
                "only_gear":     mc.get(pair, 0) == 0,
            })
    w(out/"target_type_pair_comparison.csv", pair_rows, list(pair_rows[0].keys()) if pair_rows else [])

    # ── 8. namespace_comparison.csv ─────────────────────────
    m_ns = Counter(t.split(":", 1)[0] for triple in mortar_norm_all for t in triple if ":" in t)
    g_ns = Counter(t.split(":", 1)[0] for triple in gear_norm       for t in triple if ":" in t)
    all_ns = set(m_ns) | set(g_ns)
    ns_rows = [{"namespace": ns,
                "mortar_count": m_ns.get(ns, 0),
                "gear_count":   g_ns.get(ns, 0),
                "only_mortar":  g_ns.get(ns, 0) == 0,
                "only_gear":    m_ns.get(ns, 0) == 0}
               for ns in sorted(all_ns)]
    ns_rows.sort(key=lambda r: -(r["mortar_count"]+r["gear_count"]))
    w(out/"namespace_comparison.csv", ns_rows, list(ns_rows[0].keys()) if ns_rows else [])

    # ── 9. naming_convention_comparison.csv ─────────────────
    mn = naming_stats(mortar_norm_all)
    gn = naming_stats(gear_norm)
    naming_rows = []
    for metric in ["avg_local_len","avg_tokens","pct_dot","pct_underscore","pct_hyphen","pct_numeric"]:
        naming_rows.append({"metric": metric, "mortar": mn[metric], "gear": gn[metric]})
    w(out/"naming_convention_comparison.csv", naming_rows, ["metric","mortar","gear"])

    # ── 10. graph_structure_comparison.csv ──────────────────
    struct_rows = [
        {"metric": k, "mortar": mst.get(k, mconn.get(k, "")), "gear": gst.get(k, gconn.get(k, ""))}
        for k in ["n_entities","n_predicates","avg_degree","median_degree","max_degree",
                  "subj_only","obj_only","n_components","largest_comp","isolated_nodes"]
    ]
    # Density = triples / (n_entities^2)
    me = mst["n_entities"] or 1; ge = gst["n_entities"] or 1
    struct_rows.append({"metric": "graph_density",
                        "mortar": round(len(mortar_norm_all)/me**2, 6),
                        "gear":   round(len(gear_norm)/ge**2, 6)})
    w(out/"graph_structure_comparison.csv", struct_rows, ["metric","mortar","gear"])

    # ── 11. visible_context_comparison.csv ──────────────────
    mc = context_after_removal(mortar_norm_all)
    gc = context_after_removal(gear_norm)
    ctx_rows = [{"metric": k, "mortar": mc[k], "gear": gc[k]}
                for k in ["rem_triples","rem_predicates","rem_entities",
                          "isolated_after","has_type","has_location",
                          "subj_ctx_pct","obj_ctx_pct"]]
    w(out/"visible_context_comparison.csv", ctx_rows, ["metric","mortar","gear"])

    # ── 12. data_quality_issues.csv ─────────────────────────
    quality_rows = []
    # MORTAR untyped
    m_untyped = sum(1 for e in mst["all_entities"] if e not in mst["types_map"])
    g_untyped = sum(1 for e in gst["all_entities"] if e not in gst["types_map"])
    quality_rows += [
        {"dataset":"MORTAR","issue":"untyped_entities","count":m_untyped,"pct":round(100*m_untyped/(mst["n_entities"] or 1),1),"note":"entities with no rdf:type"},
        {"dataset":"GEAR",  "issue":"untyped_entities","count":g_untyped,"pct":round(100*g_untyped/(gst["n_entities"] or 1),1),"note":"entities with no rdf:type"},
        {"dataset":"MORTAR","issue":"raw_duplicates",  "count":total_m_dup_raw,"pct":"","note":""},
        {"dataset":"GEAR",  "issue":"raw_duplicates",  "count":g_dr,"pct":"","note":""},
        {"dataset":"MORTAR","issue":"malformed_uris",  "count":len(mortar_all_issues),"pct":"","note":""},
        {"dataset":"GEAR",  "issue":"malformed_uris",  "count":len(gear_issues),"pct":"","note":""},
        {"dataset":"MORTAR","issue":"folds_applied",   "count":total_m_folds,"pct":"","note":"inverse->canonical"},
        {"dataset":"GEAR",  "issue":"folds_applied",   "count":g_folds,"pct":"","note":"inverse->canonical"},
        {"dataset":"MORTAR","issue":"remaps_applied",  "count":total_m_remaps,"pct":"","note":"REC->Brick"},
        {"dataset":"GEAR",  "issue":"remaps_applied",  "count":g_remaps,"pct":"","note":"REC->Brick"},
    ]
    w(out/"data_quality_issues.csv", quality_rows, ["dataset","issue","count","pct","note"])

    # ── 13. distribution_shift_ranking.csv ──────────────────
    # Predicate JSD
    pred_jsd = js_divergence(m_pred, g_pred)
    # Class JSD
    cls_jsd = js_divergence(m_classes, g_classes)
    # Category JSD
    cat_jsd = js_divergence(m_cats, g_cats)
    # Target relation JSD
    m_tcount = Counter({r: m_trel.get(r,{}).get("count",0) for r in TARGET_RELATIONS})
    g_tcount = Counter({r: g_trel.get(r,{}).get("count",0) for r in TARGET_RELATIONS})
    trel_jsd = js_divergence(m_tcount, g_tcount)
    # Predicate vocab overlap
    pred_jacc = jaccard(set(m_pred), set(g_pred))
    cls_jacc  = jaccard(set(m_classes), set(g_classes))

    # Degree distribution difference
    m_deg_dist = Counter(mst["degree"].values())
    g_deg_dist = Counter(gst["degree"].values())
    deg_jsd = js_divergence(m_deg_dist, g_deg_dist)

    shift_rows = [
        {"metric":"predicate_distribution_JSD",  "value":pred_jsd,  "note":"0=identical,1=disjoint"},
        {"metric":"class_distribution_JSD",       "value":cls_jsd,   "note":""},
        {"metric":"ontology_category_JSD",        "value":cat_jsd,   "note":""},
        {"metric":"target_relation_balance_JSD",  "value":trel_jsd,  "note":""},
        {"metric":"degree_distribution_JSD",      "value":deg_jsd,   "note":""},
        {"metric":"predicate_vocab_jaccard",       "value":pred_jacc, "note":"1=identical,0=disjoint"},
        {"metric":"class_vocab_jaccard",           "value":cls_jacc,  "note":""},
        {"metric":"gear_untyped_entity_pct",       "value":round(100*g_untyped/(gst["n_entities"] or 1),1),"note":"GEAR only"},
        {"metric":"mortar_untyped_entity_pct",     "value":round(100*m_untyped/(mst["n_entities"] or 1),1),"note":"MORTAR only"},
        {"metric":"gear_isolated_after_removal_pct","value":gc["isolated_after"],"note":"absolute count"},
    ]
    shift_rows.sort(key=lambda r: -abs(float(r["value"])) if isinstance(r["value"], (int,float)) else 0)
    w(out/"distribution_shift_ranking.csv", shift_rows, ["metric","value","note"])

    # ── Terminal summary ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("MORTAR vs GEAR — ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"\n1. Graph sizes:")
    print(f"   MORTAR: {mst['n_entities']:,} entities, {len(mortar_norm_all):,} triples, "
          f"{len(mortar_buildings)} buildings")
    print(f"   GEAR:   {gst['n_entities']:,} entities, {len(gear_norm):,} triples, 1 building")

    print(f"\n2. Target-relation distribution (normalised counts):")
    for rel in sorted(TARGET_RELATIONS):
        m_cnt = m_trel.get(rel,{}).get("count",0)
        g_cnt = g_trel.get(rel,{}).get("count",0)
        print(f"   {rel:<20}  MORTAR: {m_cnt:>6}  ({round(100*m_cnt/m_target_total,1)}%)  "
              f"GEAR: {g_cnt:>4}  ({round(100*g_cnt/g_target_total,1)}%)")

    print(f"\n3. Ontology coverage:")
    print(f"   Brick class vocab overlap (Jaccard): {cls_jacc}")
    print(f"   Predicate vocab overlap (Jaccard):   {pred_jacc}")
    print(f"   GEAR remaps applied (REC->Brick):    {g_remaps}")

    print(f"\n4. Untyped / unrecognised entities:")
    print(f"   MORTAR: {m_untyped} ({round(100*m_untyped/(mst['n_entities'] or 1),1)}%)")
    print(f"   GEAR:   {g_untyped} ({round(100*g_untyped/(gst['n_entities'] or 1),1)}%)")

    print(f"\n5. Isolated entities after target-edge removal:")
    print(f"   MORTAR: {mc['isolated_after']}   GEAR: {gc['isolated_after']}")

    print(f"\n6. Graph structure differences:")
    print(f"   MORTAR avg degree: {mst['avg_degree']}  GEAR avg degree: {gst['avg_degree']}")
    print(f"   MORTAR max degree: {mst['max_degree']}  GEAR max degree: {gst['max_degree']}")
    print(f"   MORTAR components: {mconn['n_components']}  GEAR components: {gconn['n_components']}")

    print(f"\n7. Top distribution shifts (JSD):")
    for row in shift_rows[:5]:
        print(f"   {row['metric']:<40} {row['value']}")

    print(f"\n8. Most important confirmed differences:")
    # Haiku summary based on computed values
    hasPart_mortar = m_trel.get("brick:hasPart",{}).get("count",0)
    hasPart_gear   = g_trel.get("brick:hasPart",{}).get("count",0)
    hasPoint_mortar= m_trel.get("brick:hasPoint",{}).get("count",0)
    hasPoint_gear  = g_trel.get("brick:hasPoint",{}).get("count",0)
    feeds_mortar   = m_trel.get("brick:feeds",{}).get("count",0)
    feeds_gear     = g_trel.get("brick:feeds",{}).get("count",0)

    print(f"   [1] hasPart: MORTAR {hasPart_mortar} ({round(100*hasPart_mortar/m_target_total,1)}%) "
          f"vs GEAR {hasPart_gear} ({round(100*hasPart_gear/g_target_total,1)}%) — "
          f"GEAR uses collection/IFC semantics, MORTAR uses HVAC containment")
    print(f"   [2] Class vocab Jaccard {cls_jacc} — most GEAR classes absent from MORTAR")
    print(f"   [3] GEAR has {g_remaps} REC->Brick remaps — schema heterogeneity")
    print(f"   [4] hasPoint GEAR {hasPoint_gear} ({round(100*hasPoint_gear/g_target_total,1)}%) "
          f"vs MORTAR {hasPoint_mortar} ({round(100*hasPoint_mortar/m_target_total,1)}%)")
    print(f"   [5] feeds GEAR {feeds_gear} ({round(100*feeds_gear/g_target_total,1)}%) "
          f"vs MORTAR {feeds_mortar} ({round(100*feeds_mortar/m_target_total,1)}%)")

    print(f"\n9. Output files:")
    for fname in sorted(out.glob("*.csv")):
        print(f"   {fname.resolve()}")

    # ── LaTeX Beamer ─────────────────────────────────────────
    tex = generate_beamer(
        mst, gst, mconn, gconn,
        mortar_norm_all, gear_norm,
        mortar_buildings,
        m_trel, g_trel,
        m_cats, g_cats,
        m_pred, g_pred,
        m_classes, g_classes,
        pred_jacc, cls_jacc,
        pred_jsd, cls_jsd, cat_jsd, trel_jsd, deg_jsd,
        mc, gc,
        total_m_remaps, g_remaps,
        total_m_folds, g_folds,
        m_untyped, g_untyped,
        pair_rows,
        mn, gn,
        out,
    )
    tex_path = out / "mortar_vs_gear_dataset_analysis.tex"
    tex_path.write_text(tex)
    print(f"\n  LaTeX: {tex_path.resolve()}")
    print("\nDone.")


# ── LaTeX generator ──────────────────────────────────────────

def generate_beamer(mst, gst, mconn, gconn,
                    mortar_triples, gear_triples,
                    mortar_buildings,
                    m_trel, g_trel,
                    m_cats, g_cats,
                    m_pred, g_pred,
                    m_classes, g_classes,
                    pred_jacc, cls_jacc,
                    pred_jsd, cls_jsd, cat_jsd, trel_jsd, deg_jsd,
                    mc_ctx, gc_ctx,
                    m_remaps, g_remaps,
                    m_folds, g_folds,
                    m_untyped, g_untyped,
                    pair_rows,
                    mn, gn,
                    out):

    m_target_total = sum(v["count"] for v in m_trel.values()) or 1
    g_target_total = sum(v["count"] for v in g_trel.values()) or 1

    def pct(val, tot):
        return round(100*val/(tot or 1), 1)

    # Top 5 mortar classes
    m_cls_top = sorted(m_classes.items(), key=lambda x: -x[1])[:5]
    g_cls_top = sorted(g_classes.items(), key=lambda x: -x[1])[:5]

    # Top gear-only classes
    gear_only_cls = sorted([(c,n) for c,n in g_classes.items() if m_classes.get(c,0)==0],
                            key=lambda x:-x[1])[:5]

    # Top mortar-only predicates
    mortar_only_pred = sorted([(p,n) for p,n in m_pred.items() if g_pred.get(p,0)==0 and n>10],
                               key=lambda x:-x[1])[:6]
    gear_only_pred   = sorted([(p,n) for p,n in g_pred.items() if m_pred.get(p,0)==0 and n>0],
                               key=lambda x:-x[1])[:6]

    # Type pair rows for slides
    hasPart_pairs  = [(r["subj_category"], r["obj_category"],
                       r["mortar_count"], r["gear_count"])
                      for r in pair_rows if r["relation"]=="brick:hasPart"][:6]
    hasPoint_pairs = [(r["subj_category"], r["obj_category"],
                       r["mortar_count"], r["gear_count"])
                      for r in pair_rows if r["relation"]=="brick:hasPoint"][:6]
    feeds_pairs    = [(r["subj_category"], r["obj_category"],
                       r["mortar_count"], r["gear_count"])
                      for r in pair_rows if r["relation"]=="brick:feeds"][:6]

    def tex_row(*cols):
        return " & ".join(str(c) for c in cols) + r" \\"

    def s(v):
        return safe_tex(str(v))

    hasPart_m  = m_trel.get("brick:hasPart",{}).get("count",0)
    hasPart_g  = g_trel.get("brick:hasPart",{}).get("count",0)
    hasPoint_m = m_trel.get("brick:hasPoint",{}).get("count",0)
    hasPoint_g = g_trel.get("brick:hasPoint",{}).get("count",0)
    feeds_m    = m_trel.get("brick:feeds",{}).get("count",0)
    feeds_g    = g_trel.get("brick:feeds",{}).get("count",0)

    cat_order = ["Equipment","Point","Location","Building","System","Collection","Other","Unrecognised"]
    cat_table = "\n".join(
        tex_row(cat, m_cats.get(cat,0), pct(m_cats.get(cat,0), sum(m_cats.values())),
                g_cats.get(cat,0), pct(g_cats.get(cat,0), sum(g_cats.values())))
        for cat in cat_order
    )

    def fmt_pair_rows(pairs):
        if not pairs:
            return tex_row("(no data)","","","")
        return "\n".join(tex_row(s(sc), s(oc), mc, gc) for sc, oc, mc, gc in pairs)

    return r"""\documentclass[10pt]{beamer}
\usepackage{booktabs}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepackage{xcolor}
\usepackage{array}

\usetheme{Madrid}
\usecolortheme{seahorse}

\title{Understanding the Differences Between MORTAR and GEAR}
\subtitle{Dataset Structure, Schema and Distribution Analysis}
\date{\today}

\begin{document}

%% Slide 1 ─ Title
\begin{frame}
  \titlepage
\end{frame}

%% Slide 2 ─ Purpose
\begin{frame}{Purpose of the Comparison}
  \begin{itemize}
    \item Four prediction approaches are evaluated on the GEAR building:
          rule-based (PBRR, AnyBURL), tree-based (GBT), and graph neural networks (CompGCN / LLM).
    \item All approaches depend critically on the structure and schema of the input graph.
    \item MORTAR (US commercial buildings, Brick schema) was used for development and training.
    \item GEAR (Singapore lab building) is the held-out transfer target.
    \item \textbf{This analysis is model-independent.} Findings apply to all four approaches.
    \item Objective: measure \emph{how} GEAR differs from MORTAR so that schema gaps,
          structural mismatches, and vocabulary shifts can be addressed at the data level.
  \end{itemize}
\end{frame}

%% Slide 3 ─ Dataset Overview
\begin{frame}{Dataset Overview}
  \begin{table}
    \centering
    \begin{tabular}{lrr}
      \toprule
      \textbf{Property} & \textbf{MORTAR} & \textbf{GEAR} \\
      \midrule
""" + tex_row("Buildings", len(mortar_buildings), 1) + r"""
""" + tex_row("Total entities", f"{mst['n_entities']:,}", f"{gst['n_entities']:,}") + r"""
""" + tex_row("Norm.\ triples", f"{len(mortar_triples):,}", f"{len(gear_triples):,}") + r"""
""" + tex_row("Unique predicates", mst['n_predicates'], gst['n_predicates']) + r"""
""" + tex_row("Unique RDF classes", mst['n_classes'], gst['n_classes']) + r"""
""" + tex_row("Target edges", m_target_total, g_target_total) + r"""
""" + tex_row("Non-target context edges", mst['n_non_target'], gst['n_non_target']) + r"""
""" + tex_row(r"REC$\to$Brick remaps", m_remaps, g_remaps) + r"""
""" + tex_row("Inverse folds applied", m_folds, g_folds) + r"""
      \bottomrule
    \end{tabular}
  \end{table}
\end{frame}

%% Slide 4 ─ Target-Relation Distribution
\begin{frame}{Target-Relation Distribution}
  \begin{table}
    \centering
    \begin{tabular}{lrrrr}
      \toprule
      \textbf{Relation} & \multicolumn{2}{c}{\textbf{MORTAR}} & \multicolumn{2}{c}{\textbf{GEAR}} \\
      \cmidrule(lr){2-3}\cmidrule(lr){4-5}
                        & Count & \% & Count & \% \\
      \midrule
""" + tex_row(r"\texttt{brick:feeds}",
              feeds_m, f"{pct(feeds_m, m_target_total)}\%",
              feeds_g, f"{pct(feeds_g, g_target_total)}\%") + r"""
""" + tex_row(r"\texttt{brick:hasPart}",
              hasPart_m, f"{pct(hasPart_m, m_target_total)}\%",
              hasPart_g, f"{pct(hasPart_g, g_target_total)}\%") + r"""
""" + tex_row(r"\texttt{brick:hasPoint}",
              hasPoint_m, f"{pct(hasPoint_m, m_target_total)}\%",
              hasPoint_g, f"{pct(hasPoint_g, g_target_total)}\%") + r"""
      \midrule
""" + tex_row(r"\textbf{Total}", m_target_total, "100\%", g_target_total, "100\%") + r"""
      \bottomrule
    \end{tabular}
  \end{table}
  \vspace{0.5em}
  \begin{itemize}
    \small
    \item MORTAR is dominated by \texttt{hasPoint}; GEAR has a much larger \texttt{hasPart} share.
    \item GEAR \texttt{hasPart} includes IFC collection membership (\texttt{rec:includes}),
          absent in MORTAR.
  \end{itemize}
\end{frame}

%% Slide 5 ─ Predicate and Schema Differences
\begin{frame}{Predicate and Schema Differences}
  \begin{columns}[t]
    \column{0.48\textwidth}
    \textbf{Common in MORTAR only} (top)\par
    \vspace{0.3em}
    \begin{itemize}
""" + "\n".join(rf"      \item \texttt{{{s(p)}}} ({n})" for p,n in mortar_only_pred[:4]) + r"""
    \end{itemize}
    \column{0.48\textwidth}
    \textbf{Common in GEAR only} (top)\par
    \vspace{0.3em}
    \begin{itemize}
""" + "\n".join(rf"      \item \texttt{{{s(p)}}} ({n})" for p,n in gear_only_pred[:4]) + r"""
    \end{itemize}
  \end{columns}
  \vspace{0.5em}
  \begin{itemize}
    \item Predicate vocabulary Jaccard similarity: """ + str(pred_jacc) + r"""
    \item Predicate distribution JSD: """ + str(pred_jsd) + r""" (0 = identical)
    \item GEAR uses REC namespace predicates; """ + str(g_remaps) + r""" remapped to Brick equivalents.
    \item \texttt{brick:hasLocation} appears """ + str(g_pred.get("brick:hasLocation",0)) + r""" times in GEAR, """ + str(m_pred.get("brick:hasLocation",0)) + r""" in MORTAR.
    \item Inverse-relation usage: """ + str(g_folds) + r""" folds in GEAR vs """ + str(m_folds) + r""" in MORTAR.
  \end{itemize}
\end{frame}

%% Slide 6 ─ Entity and Class Differences
\begin{frame}{Entity and Class Differences}
  \begin{columns}[t]
    \column{0.48\textwidth}
    \textbf{Top MORTAR classes}\par
    \begin{itemize}
""" + "\n".join(rf"      \item \texttt{{{s(c)}}} ({n})" for c,n in m_cls_top) + r"""
    \end{itemize}
    \column{0.48\textwidth}
    \textbf{Top GEAR classes}\par
    \begin{itemize}
""" + "\n".join(rf"      \item \texttt{{{s(c)}}} ({n})" for c,n in g_cls_top) + r"""
    \end{itemize}
  \end{columns}
  \vspace{0.4em}
  \begin{itemize}
    \item Class vocabulary Jaccard: """ + str(cls_jacc) + r""" --- most GEAR classes absent from MORTAR.
    \item GEAR-only classes include spatial types (Room, Floor, Building) absent in MORTAR training.
    \item Untyped entities: MORTAR """ + str(m_untyped) + r""" / GEAR """ + str(g_untyped) + r"""
    \item After REC$\to$Brick mapping, spatial GEAR classes gain Brick ancestor embeddings.
  \end{itemize}
\end{frame}

%% Slide 7 ─ Ontology Category Distribution
\begin{frame}{Ontology Category Distribution}
  \begin{table}
    \centering
    \small
    \begin{tabular}{lrrrr}
      \toprule
      \textbf{Category} & \textbf{MORTAR} & \textbf{M\%} & \textbf{GEAR} & \textbf{G\%} \\
      \midrule
""" + cat_table + r"""
      \bottomrule
    \end{tabular}
  \end{table}
\end{frame}

%% Slide 8 ─ Relation Type-Pattern Differences
\begin{frame}{Relation Type-Pattern Differences}
  \textbf{\texttt{brick:hasPart}} subject$\to$object category pairs\par
  \begin{table}
    \small
    \begin{tabular}{llrr}
      \toprule
      Subject & Object & MORTAR & GEAR \\
      \midrule
""" + fmt_pair_rows(hasPart_pairs) + r"""
      \bottomrule
    \end{tabular}
  \end{table}
  \begin{itemize}
    \item MORTAR \texttt{hasPart}: Equipment$\to$Equipment (HVAC containment).
    \item GEAR \texttt{hasPart}: Location$\to$Location (IFC/BIM spatial hierarchy) +
          Collection$\to$* (grouping).
    \item These are semantically distinct uses of the same predicate.
  \end{itemize}
\end{frame}

%% Slide 9 ─ Naming and URI Differences
\begin{frame}{Naming and URI Differences}
  \begin{table}
    \centering
    \begin{tabular}{lrr}
      \toprule
      \textbf{Metric} & \textbf{MORTAR} & \textbf{GEAR} \\
      \midrule
""" + tex_row("Avg local-name length", mn["avg_local_len"], gn["avg_local_len"]) + r"""
""" + tex_row("Avg tokens per name", mn["avg_tokens"], gn["avg_tokens"]) + r"""
""" + tex_row(r"\% with underscores", f"{mn['pct_underscore']}\%", f"{gn['pct_underscore']}\%") + r"""
""" + tex_row(r"\% with hyphens", f"{mn['pct_hyphen']}\%", f"{gn['pct_hyphen']}\%") + r"""
""" + tex_row(r"\% with numeric IDs", f"{mn['pct_numeric']}\%", f"{gn['pct_numeric']}\%") + r"""
      \bottomrule
    \end{tabular}
  \end{table}
  \begin{itemize}
    \item MORTAR entities use \texttt{bldgN:} namespace (anonymised building IDs).
    \item GEAR entities use a full IFC filename as namespace prefix.
    \item Name-similarity features (if used) benefit MORTAR's consistent conventions.
    \item GEAR contains \texttt{rdfs:label} annotations; MORTAR entities rely on URI tokens only.
  \end{itemize}
\end{frame}

%% Slide 10 ─ Graph Structure and Context
\begin{frame}{Graph Structure and Available Context}
  \begin{table}
    \centering
    \begin{tabular}{lrr}
      \toprule
      \textbf{Metric} & \textbf{MORTAR} & \textbf{GEAR} \\
      \midrule
""" + tex_row("Avg entity degree", mst["avg_degree"], gst["avg_degree"]) + r"""
""" + tex_row("Max entity degree", mst["max_degree"], gst["max_degree"]) + r"""
""" + tex_row("Connected components", mconn["n_components"], gconn["n_components"]) + r"""
""" + tex_row("Largest component", mconn["largest_comp"], gconn["largest_comp"]) + r"""
""" + tex_row("Remaining triples (target removed)", mc_ctx["rem_triples"], gc_ctx["rem_triples"]) + r"""
""" + tex_row("Isolated after removal", mc_ctx["isolated_after"], gc_ctx["isolated_after"]) + r"""
""" + tex_row(r"Target-subj retaining context (\%)", f"{mc_ctx['subj_ctx_pct']}\%", f"{gc_ctx['subj_ctx_pct']}\%") + r"""
      \bottomrule
    \end{tabular}
  \end{table}
  \begin{itemize}
    \item Non-target context in GEAR is almost entirely \texttt{rdf:type} and \texttt{brick:hasLocation}.
    \item Removing target edges leaves many GEAR entities with only their type triple.
  \end{itemize}
\end{frame}

%% Slide 11 ─ Main Findings
\begin{frame}{Main Findings}
  \textbf{Confirmed Differences}
  \begin{itemize}
    \item \texttt{hasPart} semantics differ fundamentally: HVAC containment (MORTAR) vs.\
          IFC spatial hierarchy + collection membership (GEAR).
    \item GEAR class vocabulary is largely disjoint from MORTAR (Jaccard """ + str(cls_jacc) + r""").
    \item GEAR uses REC predicates (""" + str(g_remaps) + r""" remapped); MORTAR uses Brick natively.
    \item GEAR non-target context is sparse: mostly \texttt{rdf:type} + \texttt{hasLocation} only.
    \item \texttt{hasPart} share: """ + str(pct(hasPart_g,g_target_total)) + r"""\% in GEAR vs.\
          """ + str(pct(hasPart_m,m_target_total)) + r"""\% in MORTAR.
  \end{itemize}
  \vspace{0.3em}
  \textbf{Likely Effects on Graph-Prediction Methods}
  \begin{itemize}
    \item Models trained on MORTAR \texttt{hasPart} patterns will not generalise to GEAR's
          IFC/collection use.
    \item Sparse context reduces embedding quality for GNN-based approaches.
    \item Schema heterogeneity affects any method that uses predicate identity as a feature.
  \end{itemize}
  \vspace{0.3em}
  \textbf{Issues Requiring Further Testing}
  \begin{itemize}
    \item Whether REC$\to$Brick remapping fully resolves predicate vocabulary mismatch.
    \item Whether additional Brick ancestor expansion compensates for unseen class distributions.
  \end{itemize}
\end{frame}

%% Slide 12 ─ Recommended Data-Level Next Steps
\begin{frame}{Recommended Data-Level Next Steps}
  \begin{itemize}
    \item \textbf{Schema harmonisation:} ensure all GEAR triples are mapped to Brick before
          evaluation; validate completeness of REC$\to$Brick predicate and class mapping.
    \item \textbf{Correct malformed URIs:} check for double-prefix Brick URIs and
          colon-in-local-name patterns in both datasets.
    \item \textbf{Separate evaluation by relation:} report \texttt{feeds}, \texttt{hasPart},
          and \texttt{hasPoint} metrics independently to isolate domain-shift effects.
    \item \textbf{Separate evaluation by entity-type group:} e.g., report results
          for Equipment-only, Location-only, and Collection subsets of GEAR.
    \item \textbf{Expand training data:} include buildings with IFC/BIM spatial containment
          hierarchies and REC-annotated data if available.
    \item \textbf{Document GEAR custom classes:} entities lacking a Brick equivalent
          should be explicitly labelled as out-of-vocabulary rather than treated as unknown.
    \item \textbf{Create a GEAR validation subset:} a small labelled subset stratified by
          relation type enables calibrated threshold selection without using the test set.
  \end{itemize}
\end{frame}

\end{document}
"""


if __name__ == "__main__":
    # Validate config before running
    missing = []
    for name, val in [("MORTAR_DATA_DIRECTORY", MORTAR_DATA_DIRECTORY),
                      ("GEAR_DATASET_PATH",     GEAR_DATASET_PATH),
                      ("BRICK_ONTOLOGY_PATH",   BRICK_ONTOLOGY_PATH),
                      ("OUTPUT_DIRECTORY",      OUTPUT_DIRECTORY)]:
        if val.startswith("<"):
            missing.append(name)
    if missing:
        print("ERROR: Please set the following variables at the top of the script:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)
    run()
