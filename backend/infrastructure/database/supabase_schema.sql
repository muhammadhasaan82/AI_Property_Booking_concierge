-- ============================================================
-- Hotel Booking - Supabase Schema
-- Run this in Supabase SQL Editor.
-- ============================================================

create extension if not exists "pgcrypto";

-- users
create table if not exists public.users (
    id uuid primary key default gen_random_uuid(),
    name text,
    email text unique not null,
    phone text,
    created_at timestamptz not null default now()
);

create unique index if not exists users_email_idx on public.users(email);

-- bookings (core transactional table)
create table if not exists public.bookings (
    id uuid primary key default gen_random_uuid(),
    user_id uuid references public.users(id),
    property_id text,
    check_in date,
    check_out date,
    guests integer default 1,
    phone text,
    status text default 'pending',
    payment_url text,
    created_at timestamptz not null default now()
);

-- chat history (all user/bot turns)
create table if not exists public.chat_history (
    id bigserial primary key,
    user_message text not null,
    bot_response text not null,
    created_at timestamptz not null default now()
);

-- successful bookings only (for status checks in chat)
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
    guests integer,
    nights integer,
    total_amount numeric(12,2),
    payment_url text,
    source text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists successful_bookings_user_email_idx
    on public.successful_bookings(user_email);

-- Optional RLS for local/dev usage:
-- alter table public.users enable row level security;
-- alter table public.bookings enable row level security;
-- alter table public.chat_history enable row level security;
-- alter table public.successful_bookings enable row level security;
-- create policy "anon_all_users" on public.users for all using (true) with check (true);
-- create policy "anon_all_bookings" on public.bookings for all using (true) with check (true);
-- create policy "anon_all_chat_history" on public.chat_history for all using (true) with check (true);
-- create policy "anon_all_successful_bookings" on public.successful_bookings for all using (true) with check (true);
