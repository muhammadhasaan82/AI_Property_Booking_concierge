-- Add per-night price to durable booking receipts used by cross-session status lookup.
alter table public.successful_bookings
    add column if not exists price_per_night numeric(12,2);
