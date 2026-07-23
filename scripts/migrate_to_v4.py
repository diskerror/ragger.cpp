#!/usr/bin/env python3
"""
Migrate Ragger database from v3 to v4 schema.

*** REFERENCE ONLY — NOT the live migration path (as of v0.11.0). ***

The v3->v4 migration is now performed IN-PROCESS by the C++ backend
(src/sqlite_backend.cpp: migrate_schema_to_v4 / rebuild_tables_v3_to_v4). Any
daemon >= v0.11.0 detects a pre-v4 DB (settings['db_version'] absent AND
models.created_at missing) and reshapes it automatically on open, then stamps
settings['db_version'] = 4. You do NOT need to run this script against a live
Ragger install — starting the v0.11.0+ daemon is sufficient and is the tested,
sanctioned path.

This script is kept only as:
  * an offline / rescue tool for migrating a DB *copy* without building or
    running the daemon (e.g. inspecting an old backup on a machine without the
    C++ toolchain), and
  * a readable, self-contained reference for exactly what the v4 reshape does.

CAUTION: with default args it operates on ~/.ragger/memories.db via an
in-place same-path swap. Never run it against the database of a RUNNING daemon
(WAL contention can corrupt state) — stop the daemon first, or point --db at a
throwaway copy.

This script:
1. Creates a backup of the current DB (memories.db -> memories.db.v3.backup)
2. Creates a new DB with v4 schema
3. Migrates data preserving timestamps
4. Adds created_at to models/sessions if missing (using existing row creation time approx)

Usage:
    python3 migrate_to_v4.py [--db PATH] [--dry-run]

Defaults to ~/.ragger/memories.db
"""

import sqlite3
import os
import sys
import shutil
from datetime import datetime
from pathlib import Path


def get_timestamp():
    """Return current timestamp in SQLite-compatible format."""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def read_schema(path):
    """Read SQL schema file."""
    with open(path, 'r') as f:
        return f.read()


def migrate_db(source_path, dest_path, dry_run=False):
    """Perform the migration from v3 to v4 schema."""
    
    print(f"Source DB: {source_path}")
    print(f"Dest DB:   {dest_path}")
    if dry_run:
        print("DRY RUN - no changes will be made")
    
    # Verify source exists
    if not os.path.exists(source_path):
        print(f"ERROR: Source database not found: {source_path}")
        sys.exit(1)
    
    # Create backup of source
    backup_path = str(source_path) + '.v3.backup'
    if not dry_run:
        print(f"\nCreating backup: {backup_path}")
        shutil.copy2(source_path, backup_path)
        print("Backup complete.")
    
    # Read v4 schema
    script_dir = Path(__file__).parent
    schema_path = script_dir / 'schema_v4.sql'
    if not schema_path.exists():
        print(f"ERROR: Schema file not found: {schema_path}")
        sys.exit(1)
    
    v4_schema = read_schema(schema_path)
    
    if dry_run:
        print("\nWould create new DB with v4 schema")
        print("Would migrate data from source to destination")
        return
    
    # If dest is the same as source, we need a temp file for the new DB
    temp_path = None
    if os.path.abspath(source_path) == os.path.abspath(dest_path):
        temp_path = dest_path + '.v4.new'
        print(f"\nUsing temp file: {temp_path}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        effective_dest = temp_path
    else:
        # Remove dest if it exists (clean slate)
        if os.path.exists(dest_path):
            os.remove(dest_path)
            print(f"Removed existing destination: {dest_path}")
        effective_dest = dest_path
    
    # Create new DB with v4 schema
    print("\nCreating new database with v4 schema...")
    dest_db = sqlite3.connect(effective_dest)
    dest_db.execute('pragma encoding="UTF-8"')
    dest_db.executescript(v4_schema)
    dest_db.commit()
    print("v4 schema created.")
    
    # Open source for reading
    source_db = sqlite3.connect(source_path)
    source_db.row_factory = None
    
    # Check if models/sessions have created_at already
    def table_has_column(db, table, col):
        result = db.execute(
            f"SELECT COUNT(*) FROM pragma_table_info('{table}') WHERE name=?", 
            (col,)
        )
        return result.fetchone()[0] > 0
    
    models_have_created = table_has_column(source_db, 'models', 'created_at')
    sessions_have_created = table_has_column(source_db, 'sessions', 'created_at')
    
    print(f"\nSource DB status:")
    print(f"  models has created_at: {models_have_created}")
    print(f"  sessions has created_at: {sessions_have_created}")
    
    # Migrate each table
    
    # 1. users
    print("\nMigrating users...")
    rows = list(source_db.execute(
        "SELECT id, username, token_hash, password_hash, created, modified FROM users"
    ))
    dest_db.executemany(
        "INSERT INTO users (id, username, token_hash, password_hash, created, modified) VALUES (?, ?, ?, ?, ?, ?)",
        rows
    )
    print(f"  Copied {len(rows)} users")
    
    # 2. models - need to handle missing created_at
    print("\nMigrating models...")
    if models_have_created:
        rows = list(source_db.execute(
            "SELECT model_id, name, created_at FROM models"
        ))
        dest_db.executemany(
            "INSERT INTO models (model_id, name, created_at) VALUES (?, ?, ?)",
            rows
        )
    else:
        # No created_at in source - use a placeholder (epoch or migration time)
        rows = list(source_db.execute("SELECT model_id, name FROM models"))
        migration_time = get_timestamp()
        dest_db.executemany(
            "INSERT INTO models (model_id, name, created_at) VALUES (?, ?, ?)",
            [(mid, name, migration_time) for mid, name in rows]
        )
    print(f"  Copied {len(rows)} models")
    
    # 3. sessions - need to handle missing created_at  
    print("\nMigrating sessions...")
    if sessions_have_created:
        rows = list(source_db.execute(
            "SELECT session_id, guid, created_at FROM sessions"
        ))
        dest_db.executemany(
            "INSERT INTO sessions (session_id, guid, created_at) VALUES (?, ?, ?)",
            rows
        )
    else:
        # No created_at in source - use earliest turn's created_at or migration time
        # Try to find a reasonable timestamp from related turns
        earliest = source_db.execute(
            "SELECT MIN(created_at) FROM turns"
        ).fetchone()[0]
        
        if earliest:
            fallback_time = earliest
            print(f"  Using earliest turn timestamp: {fallback_time}")
        else:
            fallback_time = get_timestamp()
            print(f"  No turns found, using current time: {fallback_time}")
            
        rows = list(source_db.execute("SELECT session_id, guid FROM sessions"))
        dest_db.executemany(
            "INSERT INTO sessions (session_id, guid, created_at) VALUES (?, ?, ?)",
            [(sid, guid, fallback_time) for sid, guid in rows]
        )
    print(f"  Copied {len(rows)} sessions")
    
    # 4. turns - column order changed but we use explicit names
    print("\nMigrating turns...")
    rows = list(source_db.execute(
        "SELECT turn_id, user_text, assistant_text, model_id, session_id, created_at, embedding, phon FROM turns"
    ))
    dest_db.executemany(
        "INSERT INTO turns (turn_id, user_text, assistant_text, model_id, session_id, created_at, embedding, phon) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows
    )
    print(f"  Copied {len(rows)} turns")
    
    # 5. summaries - column order changed but we use explicit names
    print("\nMigrating summaries...")
    rows = list(source_db.execute(
        "SELECT summary_id, text, level, status, tags, session_id, model_id, created_at, updated_at, embedding, phon FROM summaries"
    ))
    dest_db.executemany(
        "INSERT INTO summaries (summary_id, text, level, status, tags, session_id, model_id, created_at, updated_at, embedding, phon) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows
    )
    print(f"  Copied {len(rows)} summaries")
    
    # 6. decisions - column order changed but we use explicit names
    print("\nMigrating decisions...")
    rows = list(source_db.execute(
        "SELECT decision_id, text, status, tags, created_at, embedding, phon FROM decisions"
    ))
    dest_db.executemany(
        "INSERT INTO decisions (decision_id, text, status, tags, created_at, embedding, phon) VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows
    )
    print(f"  Copied {len(rows)} decisions")
    
    # 7. documents - column order changed but we use explicit names
    print("\nMigrating documents...")
    rows = list(source_db.execute(
        "SELECT document_id, text, path, title, tags, year, chunk_index, imported_at, embedding, phon FROM documents"
    ))
    dest_db.executemany(
        "INSERT INTO documents (document_id, text, path, title, tags, year, chunk_index, imported_at, embedding, phon) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows
    )
    print(f"  Copied {len(rows)} documents")
    
    # Copy settings table data (schema_v4.sql now creates the table itself).
    print("\nMigrating settings...")
    try:
        rows = list(source_db.execute("SELECT key, value FROM settings"))
        dest_db.executemany(
            "INSERT INTO settings (key, value) VALUES (?, ?)",
            rows
        )
        print(f"  Copied {len(rows)} settings")
    except sqlite3.OperationalError:
        print("  No settings table found (ok for fresh installs)")
    
    dest_db.commit()
    
    # Verify counts match
    print("\n=== Verification ===")
    for table in ['users', 'models', 'sessions', 'turns', 'summaries', 'decisions', 'documents']:
        src_count = source_db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        dst_count = dest_db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        status = "✓" if src_count == dst_count else "✗ MISMATCH"
        print(f"{table:15s}: source={src_count:,}  dest={dst_count:,}  {status}")
    
    # Verify schema
    print("\n=== Destination Schema ===")
    tables = dest_db.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE '%_fts%' ORDER BY name"
    )
    for name, sql in tables:
        if sql:
            # Extract first few lines of schema
            lines = sql.split('\n')[:5]
            print(f"\n{name}:")
            for line in lines:
                print(f"  {line.strip()}")
    
    source_db.close()
    dest_db.close()
    
    # If we used a temp file, swap it into place
    if temp_path:
        print(f"\nSwapping new DB into place...")
        os.rename(temp_path, dest_path)
        print("Swap complete.")
    
    print("\n=== Migration Complete ===")
    print(f"Backup saved to: {backup_path}")
    print(f"New DB at:       {dest_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Migrate Ragger database to v4 schema')
    parser.add_argument('--db', type=str, help='Path to memories.db (default: ~/.ragger/memories.db)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    
    args = parser.parse_args()
    
    # Default paths
    ragger_dir = os.path.expanduser('~/.ragger')
    source_path = args.db or (ragger_dir + '/memories.db')
    
    # Destination is same as source (in-place replacement after backup)
    dest_path = source_path
    
    migrate_db(source_path, dest_path, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
