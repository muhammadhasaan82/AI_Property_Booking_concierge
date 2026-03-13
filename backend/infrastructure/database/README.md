# database/

All database-related assets for the Hotel Booking platform.

```
database/
├── migrations/
│   └── 001_initial_schema.sql   ← Run this ONCE in Supabase SQL Editor
└── rust_db_gateway/             ← Rust microservice for all DB operations
    ├── Cargo.toml
    └── src/
        └── main.rs
```

## Setup (one-time)

### 1 — Run the SQL migration
Open [Supabase SQL Editor](https://supabase.com/dashboard/project/mdehkfdilygmscdbowbh/sql/new), paste and run `migrations/001_initial_schema.sql`.

### 2 — Set your database password
In `services/.env`, replace `YOUR_DATABASE_PASSWORD_HERE`:
```env
SUPABASE_DB_PASSWORD=your_actual_db_password
```

### 3 — Build & start the Rust DB Gateway
```bash
cd database/rust_db_gateway
cargo build --release
cargo run --release
```
Default port: **3002**

## Architecture

```
User
 │
 ▼
Python chatbot (booking.py)
 │
 ├─── POST /users/upsert   ──►  Rust DB Gateway :3002
 ├─── POST /bookings/create ──► Rust DB Gateway :3002   ──►  Supabase PostgreSQL
 ├─── GET  /bookings/:id    ──► Rust DB Gateway :3002
 └─── PATCH /bookings/:id   ──► Rust DB Gateway :3002
       │
       └─ (fallback if :3002 is down)
            Supabase REST API :443 directly
```
