from __future__ import annotations
import os
from typing import Optional

import psycopg


SCHEMA_SQL = r"""
-- Enable extension
create extension if not exists "pgcrypto";

-- Enum for booking status
do $$
begin
  if not exists (select 1 from pg_type where typname = 'booking_status') then
    create type booking_status as enum ('pending','confirmed','checked_in','checked_out');
  end if;
end$$;

-- Users table
create table if not exists public.users (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  email text not null unique,
  phone text,
  created_at timestamptz not null default now()
);

-- Bookings table
create table if not exists public.bookings (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references public.users(id) on delete cascade,
  property_id text not null,
  check_in date not null,
  check_out date not null,
  guests int not null default 1,
  phone text,
  status booking_status not null default 'confirmed',
  payment_url text,
  created_at timestamptz not null default now()
);

-- Chat history table
create table if not exists public.chat_history (
  id bigserial primary key,
  user_message text not null,
  bot_response text not null,
  created_at timestamptz not null default now()
);

-- Booking details table
create table if not exists public.booking_details (
  id uuid primary key default gen_random_uuid(),
  booking_id uuid not null references public.bookings(id) on delete cascade,
  booking_code text,
  property_type text,
  property_description text,
  client_name text,
  client_phone text,
  client_email text,
  check_in date,
  check_out date,
  guests int,
  nights int,
  total_amount numeric(12,2),
  payment text check (payment in ('TRUE','FALSE','pending')) default 'pending',
  created_at timestamptz not null default now()
);

-- Successful bookings table (for chat status lookups)
create table if not exists public.successful_bookings (
  booking_id text primary key,
  status text not null default 'confirmed',
  check_in date,
  check_out date,
  user_name text,
  user_email text,
  user_phone text,
  property_title text,
  property_type text,
  city text,
  guests int,
  nights int,
  total_amount numeric(12,2),
  payment_url text,
  source text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- RLS and permissive policies for local dev (optional)
alter table public.users enable row level security;
alter table public.bookings enable row level security;
alter table public.chat_history enable row level security;
alter table public.booking_details enable row level security;
alter table public.successful_bookings enable row level security;

do $$
begin
  if not exists (
    select 1 from pg_policies 
    where policyname = 'allow_all_users_dev' and schemaname = 'public' and tablename = 'users'
  ) then
    create policy "allow_all_users_dev" on public.users
      for all using (true) with check (true);
  end if;
  if not exists (
    select 1 from pg_policies 
    where policyname = 'allow_all_bookings_dev' and schemaname = 'public' and tablename = 'bookings'
  ) then
    create policy "allow_all_bookings_dev" on public.bookings
      for all using (true) with check (true);
  end if;
  if not exists (
    select 1 from pg_policies 
    where policyname = 'allow_all_chat_history_dev' and schemaname = 'public' and tablename = 'chat_history'
  ) then
    create policy "allow_all_chat_history_dev" on public.chat_history
      for all using (true) with check (true);
  end if;
  if not exists (
    select 1 from pg_policies 
    where policyname = 'allow_all_booking_details_dev' and schemaname = 'public' and tablename = 'booking_details'
  ) then
    create policy "allow_all_booking_details_dev" on public.booking_details
      for all using (true) with check (true);
  end if;
  if not exists (
    select 1 from pg_policies
    where policyname = 'allow_all_successful_bookings_dev' and schemaname = 'public' and tablename = 'successful_bookings'
  ) then
    create policy "allow_all_successful_bookings_dev" on public.successful_bookings
      for all using (true) with check (true);
  end if;
end$$;
"""


def _build_conninfo() -> Optional[str]:
    """Build a Postgres conninfo string from Supabase env.
    Prefer SUPABASE_DB_URL if present; otherwise compose from known local defaults.
    """
    # If user provided a direct Postgres connection string, prefer it
    db_url = os.getenv("SUPABASE_DB_URL") or os.getenv("POSTGRES_URL")
    if db_url:
        return db_url

    # Compose from local supabase defaults
    host = os.getenv("SUPABASE_DB_HOST", "127.0.0.1")
    port = os.getenv("SUPABASE_DB_PORT", "54322")
    dbname = os.getenv("SUPABASE_DB_NAME", "postgres")
    user = os.getenv("SUPABASE_DB_USER", "postgres")
    password = os.getenv("SUPABASE_DB_PASSWORD", "postgres")

    if not host or not port or not dbname or not user:
        return None
    return f"host={host} port={port} dbname={dbname} user={user} password={password}"


def init_schema(conninfo: Optional[str] = None) -> None:
    """Create tables/policies if missing. Safe to run multiple times."""
    conninfo = conninfo or _build_conninfo()
    if not conninfo:
        raise RuntimeError("Database connection info not found. Set SUPABASE_DB_URL or SUPABASE_DB_* envs.")
    with psycopg.connect(conninfo, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)


def verify(conninfo: Optional[str] = None) -> dict:
    """Return counts and a simple health snapshot for users/bookings."""
    conninfo = conninfo or _build_conninfo()
    if not conninfo:
        raise RuntimeError("Database connection info not found. Set SUPABASE_DB_URL or SUPABASE_DB_* envs.")
    out: dict = {"ok": True}
    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute("select count(*) from public.users;")
            users = cur.fetchone()[0]
            cur.execute("select count(*) from public.bookings;")
            bookings = cur.fetchone()[0]
            # Optional tables (may not exist if older schema); guard each
            chat = None
            details = None
            successful = None
            try:
                cur.execute("select count(*) from public.chat_history;")
                chat = cur.fetchone()[0]
            except Exception:
                chat = None
            try:
                cur.execute("select count(*) from public.booking_details;")
                details = cur.fetchone()[0]
            except Exception:
                details = None
            try:
                cur.execute("select count(*) from public.successful_bookings;")
                successful = cur.fetchone()[0]
            except Exception:
                successful = None
            out.update({
                "users": users,
                "bookings": bookings,
                "chat_history": chat,
                "booking_details": details,
                "successful_bookings": successful,
            })
    return out


if __name__ == "__main__":
    import argparse, sys
    p = argparse.ArgumentParser(description="Initialize and verify Supabase DB schema.")
    p.add_argument("action", choices=["init","verify"], help="init schema or verify counts")
    p.add_argument("--conn", dest="conninfo", default=None, help="Optional Postgres conninfo")
    args = p.parse_args()

    try:
        if args.action == "init":
            init_schema(args.conninfo)
            print("Schema initialized.")
        else:
            print(verify(args.conninfo))
    except Exception as e:
        print({"ok": False, "error": str(e)})
        sys.exit(1)


