"""
src/graph/neo4j_connector.py
============================
Optional Neo4j export utilities for the mule detection graph.

This module keeps the core pipeline usable without Neo4j installed while
making it easy to push the enriched graph into a graph database for analyst
queries and graph-native exploration.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import networkx as nx
import pandas as pd

from config.config import *

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    GraphDatabase = None
    NEO4J_AVAILABLE = False


def get_neo4j_settings(
    uri: str | None = None,
    user: str | None = None,
    password: str | None = None,
    database: str | None = None,
) -> dict:
    """Resolve Neo4j settings from args first, then env-backed config."""
    return {
        "uri": uri or NEO4J_URI,
        "user": user or NEO4J_USER,
        "password": password if password is not None else NEO4J_PASSWORD,
        "database": database or NEO4J_DATABASE,
    }


def _chunked(records: list[dict], batch_size: int):
    for i in range(0, len(records), batch_size):
        yield records[i:i + batch_size]


def export_graph_to_neo4j(
    G: nx.DiGraph,
    feature_matrix: pd.DataFrame | None = None,
    communities: pd.DataFrame | None = None,
    uri: str | None = None,
    user: str | None = None,
    password: str | None = None,
    database: str | None = None,
    batch_size: int = NEO4J_BATCH_SIZE,
) -> dict:
    """
    Export the summary graph to Neo4j.

    Nodes:
      (:Account:MuleDetection {account_id, is_fraud, ...selected feature attrs})

    Relationships:
      (:Account)-[:TRANSFER {tx_count, total_amount, ...}]->(:Account)

    Community metadata, when available, is attached to Account nodes.
    """
    if not NEO4J_AVAILABLE:
        raise ImportError(
            "neo4j driver not installed. Install with: python3 -m pip install neo4j"
        )

    settings = get_neo4j_settings(uri=uri, user=user, password=password, database=database)
    if not settings["password"]:
        raise ValueError("Neo4j password is required. Set NEO4J_PASSWORD or pass --neo4j-password.")

    driver = GraphDatabase.driver(
        settings["uri"],
        auth=(settings["user"], settings["password"]),
    )

    fm = None
    if feature_matrix is not None:
        fm = feature_matrix.copy()
        if "account" in fm.columns:
            fm = fm.set_index("account")
        fm = fm.where(pd.notnull(fm), None)

    community_lookup = {}
    if communities is not None and len(communities) > 0:
        comm_cols = [c for c in communities.columns if c != "accounts"]
        for row in communities[comm_cols].to_dict(orient="records"):
            community_lookup[row["community_id"]] = row

    node_rows = []
    for node, attrs in G.nodes(data=True):
        props = {
            "account_id": node,
            "is_fraud": int(attrs.get("is_fraud", 0)),
        }
        if fm is not None and node in fm.index:
            fm_row = fm.loc[node].to_dict()
            for key in [
                "community_id", "tx_count", "total_sent", "total_recv",
                "pagerank", "betweenness", "burst_ratio", "risk_level",
            ]:
                if key in fm_row:
                    props[key] = fm_row[key]

            community_id = fm_row.get("community_id")
            if community_id in community_lookup:
                comm = community_lookup[community_id]
                props["community_suspicion_score"] = comm.get("suspicion_score")
                props["community_is_suspicious"] = comm.get("is_suspicious")
                props["community_size"] = comm.get("size")

        node_rows.append(props)

    edge_rows = []
    for src, dst, attrs in G.edges(data=True):
        edge_rows.append({
            "src": src,
            "dst": dst,
            "tx_count": int(attrs.get("tx_count", 0)),
            "total_amount": float(attrs.get("total_amount", 0.0)),
            "avg_amount": float(attrs.get("avg_amount", 0.0)),
            "fraud_count": int(attrs.get("fraud_count", 0)),
            "is_fraud_edge": int(attrs.get("is_fraud_edge", 0)),
            "first_seen": str(attrs.get("first_seen")) if attrs.get("first_seen") is not None else None,
            "last_seen": str(attrs.get("last_seen")) if attrs.get("last_seen") is not None else None,
        })

    with driver.session(database=settings["database"]) as session:
        session.run(
            "CREATE CONSTRAINT mule_account_id_unique IF NOT EXISTS "
            "FOR (a:MuleDetection) REQUIRE a.account_id IS UNIQUE"
        )

        node_query = """
        UNWIND $rows AS row
        MERGE (a:Account:MuleDetection {account_id: row.account_id})
        SET a += row
        """
        edge_query = """
        UNWIND $rows AS row
        MATCH (src:Account:MuleDetection {account_id: row.src})
        MATCH (dst:Account:MuleDetection {account_id: row.dst})
        MERGE (src)-[r:TRANSFER {src: row.src, dst: row.dst}]->(dst)
        SET r += row
        """

        for chunk in _chunked(node_rows, batch_size):
            session.run(node_query, rows=chunk)
        for chunk in _chunked(edge_rows, batch_size):
            session.run(edge_query, rows=chunk)

    driver.close()

    return {
        "uri": settings["uri"],
        "database": settings["database"],
        "nodes_exported": len(node_rows),
        "edges_exported": len(edge_rows),
    }
