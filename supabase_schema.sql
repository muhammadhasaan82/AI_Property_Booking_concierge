-- ============================================================
-- Hotel Booking — Supabase Schema
-- Run this ONCE in your Supabase project SQL Editor:
--   https://supabase.com/dashboard → SQL Editor → New Query
-- ============================================================

-- ── users ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.users (
    id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name      TEXT,
    email     TEXT UNIQUE NOT NULL,
    phone     TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Allow upsert by email (booking.py uses merge-duplicates)
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

-- ── Row Level Security (optional but recommended) ─────────
-- Uncomment only if you enable RLS in Supabase dashboard.
-- ALTER TABLE public.users   ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE public.bookings ENABLE ROW LEVEL SECURITY;
-- CREATE POLICY "anon_all" ON public.users   FOR ALL USING (true) WITH CHECK (true);
-- CREATE POLICY "anon_all" ON public.bookings FOR ALL USING (true) WITH CHECK (true);
