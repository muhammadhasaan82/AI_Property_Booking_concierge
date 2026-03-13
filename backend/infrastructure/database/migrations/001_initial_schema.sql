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

-- all chat turns
CREATE TABLE IF NOT EXISTS public.chat_history (
    id          BIGSERIAL PRIMARY KEY,
    user_message TEXT NOT NULL,
    bot_response TEXT NOT NULL,
    created_at   TIMESTAMPTZ DEFAULT now()
);

-- successful bookings only (used for booking status checks in chat)
CREATE TABLE IF NOT EXISTS public.successful_bookings (
    booking_id    TEXT PRIMARY KEY,
    status        TEXT DEFAULT 'confirmed',
    check_in      DATE,
    check_out     DATE,
    user_name     TEXT,
    user_email    TEXT,
    user_phone    TEXT,
    property_title TEXT,
    property_type TEXT,
    city          TEXT,
    guests        INTEGER,
    nights        INTEGER,
    total_amount  NUMERIC(12,2),
    payment_url   TEXT,
    source        TEXT,
    created_at    TIMESTAMPTZ DEFAULT now(),
    updated_at    TIMESTAMPTZ DEFAULT now()
);
