-- ============================================================
-- Hotel Booking — Supabase Schema  (v1 initial migration)
-- Run once in: Supabase Dashboard → SQL Editor → New Query
-- https://supabase.com/dashboard/project/mdehkfdilygmscdbowbh/sql/new
-- ============================================================

-- ── users ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.users (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name       TEXT,
    email      TEXT UNIQUE NOT NULL,
    phone      TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS users_email_idx ON public.users(email);

-- ── bookings ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.bookings (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      UUID REFERENCES public.users(id),
    property_id  TEXT,
    check_in     DATE,
    check_out    DATE,
    guests       INTEGER DEFAULT 1,
    phone        TEXT,
    status       TEXT DEFAULT 'pending',
    payment_url  TEXT,
    created_at   TIMESTAMPTZ DEFAULT now()
);
