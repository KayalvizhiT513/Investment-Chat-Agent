# 📊 Supabase Database Schema

This document describes all the required tables and their schemas for the Investment Analytics AI Agent.

---

## 🗄️ Required Tables

The application requires **4 tables** in your Supabase database:

1. **portfolios** - Portfolio master data
2. **portfolio_returns** - Monthly portfolio return data
3. **benchmarks** - Benchmark master data
4. **benchmark_returns** - Monthly benchmark return data

---

## 📋 Table Schemas

### 1. `portfolios` Table

**Purpose:** Stores portfolio master information

| Column Name | Data Type | Constraints | Description |
|-------------|-----------|-------------|-------------|
| `id` | `integer` | PRIMARY KEY, AUTO INCREMENT | Unique portfolio identifier |
| `portfolio_name` | `text` | NOT NULL, UNIQUE | Name of the portfolio |
| `description` | `text` | NULLABLE | Optional portfolio description |
| `created_at` | `timestamp` | DEFAULT now() | Record creation timestamp |

**SQL Creation Script:**
```sql
CREATE TABLE portfolios (
    id SERIAL PRIMARY KEY,
    portfolio_name TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

**Sample Data:**
```sql
INSERT INTO portfolios (portfolio_name, description) VALUES
('Growth Plus', 'Aggressive growth portfolio focused on high-growth equities'),
('Global Dividend', 'Income-focused portfolio with global dividend stocks'),
('Secure Income', 'Conservative fixed-income portfolio'),
('Global Macro Opportunities', 'Tactical allocation based on macro trends');
```

---

### 2. `portfolio_returns` Table

**Purpose:** Stores monthly return data for portfolios

| Column Name | Data Type | Constraints | Description |
|-------------|-----------|-------------|-------------|
| `id` | `integer` | PRIMARY KEY, AUTO INCREMENT | Unique record identifier |
| `portfolio_id` | `integer` | FOREIGN KEY → portfolios(id), NOT NULL | Reference to portfolio |
| `month_end_date` | `date` | NOT NULL | Month-end date (YYYY-MM-DD) |
| `portfolio_return` | `numeric` | NOT NULL | Monthly return (decimal, e.g., 0.05 for 5%) |
| `created_at` | `timestamp` | DEFAULT now() | Record creation timestamp |

**Constraints:**
- UNIQUE constraint on (portfolio_id, month_end_date) to prevent duplicate entries

**SQL Creation Script:**
```sql
CREATE TABLE portfolio_returns (
    id SERIAL PRIMARY KEY,
    portfolio_id INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    month_end_date DATE NOT NULL,
    portfolio_return NUMERIC NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(portfolio_id, month_end_date)
);

-- Create index for faster queries
CREATE INDEX idx_portfolio_returns_portfolio_id ON portfolio_returns(portfolio_id);
CREATE INDEX idx_portfolio_returns_date ON portfolio_returns(month_end_date);
```

**Sample Data:**
```sql
-- Example: Growth Plus portfolio returns for 2023
INSERT INTO portfolio_returns (portfolio_id, month_end_date, portfolio_return) VALUES
(1, '2023-01-31', 0.0234),  -- 2.34% return
(1, '2023-02-28', -0.0156), -- -1.56% return
(1, '2023-03-31', 0.0412),  -- 4.12% return
(1, '2023-04-30', 0.0189),  -- 1.89% return
(1, '2023-05-31', 0.0267),  -- 2.67% return
(1, '2023-06-30', 0.0345),  -- 3.45% return
(1, '2023-07-31', 0.0198),  -- 1.98% return
(1, '2023-08-31', -0.0223), -- -2.23% return
(1, '2023-09-30', 0.0156),  -- 1.56% return
(1, '2023-10-31', 0.0289),  -- 2.89% return
(1, '2023-11-30', 0.0401),  -- 4.01% return
(1, '2023-12-31', 0.0334);  -- 3.34% return
```

---

### 3. `benchmarks` Table

**Purpose:** Stores benchmark master information

| Column Name | Data Type | Constraints | Description |
|-------------|-----------|-------------|-------------|
| `id` | `integer` | PRIMARY KEY, AUTO INCREMENT | Unique benchmark identifier |
| `benchmark_name` | `text` | NOT NULL, UNIQUE | Name of the benchmark |
| `description` | `text` | NULLABLE | Optional benchmark description |
| `created_at` | `timestamp` | DEFAULT now() | Record creation timestamp |

**SQL Creation Script:**
```sql
CREATE TABLE benchmarks (
    id SERIAL PRIMARY KEY,
    benchmark_name TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

**Sample Data:**
```sql
INSERT INTO benchmarks (benchmark_name, description) VALUES
('S&P 500', 'Standard & Poor''s 500 Index'),
('MSCI World', 'MSCI World Index - Global equity benchmark'),
('Secure Income Benchmark', 'Custom fixed-income benchmark'),
('Bloomberg Barclays Aggregate', 'US investment-grade bond index');
```

---

### 4. `benchmark_returns` Table

**Purpose:** Stores monthly return data for benchmarks

| Column Name | Data Type | Constraints | Description |
|-------------|-----------|-------------|-------------|
| `id` | `integer` | PRIMARY KEY, AUTO INCREMENT | Unique record identifier |
| `benchmark_id` | `integer` | FOREIGN KEY → benchmarks(id), NOT NULL | Reference to benchmark |
| `month_end_date` | `date` | NOT NULL | Month-end date (YYYY-MM-DD) |
| `benchmark_return` | `numeric` | NOT NULL | Monthly return (decimal, e.g., 0.05 for 5%) |
| `created_at` | `timestamp` | DEFAULT now() | Record creation timestamp |

**Constraints:**
- UNIQUE constraint on (benchmark_id, month_end_date) to prevent duplicate entries

**SQL Creation Script:**
```sql
CREATE TABLE benchmark_returns (
    id SERIAL PRIMARY KEY,
    benchmark_id INTEGER NOT NULL REFERENCES benchmarks(id) ON DELETE CASCADE,
    month_end_date DATE NOT NULL,
    benchmark_return NUMERIC NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(benchmark_id, month_end_date)
);

-- Create index for faster queries
CREATE INDEX idx_benchmark_returns_benchmark_id ON benchmark_returns(benchmark_id);
CREATE INDEX idx_benchmark_returns_date ON benchmark_returns(month_end_date);
```

**Sample Data:**
```sql
-- Example: S&P 500 returns for 2023
INSERT INTO benchmark_returns (benchmark_id, month_end_date, benchmark_return) VALUES
(1, '2023-01-31', 0.0212),  -- 2.12% return
(1, '2023-02-28', -0.0134), -- -1.34% return
(1, '2023-03-31', 0.0389),  -- 3.89% return
(1, '2023-04-30', 0.0167),  -- 1.67% return
(1, '2023-05-31', 0.0245),  -- 2.45% return
(1, '2023-06-30', 0.0323),  -- 3.23% return
(1, '2023-07-31', 0.0178),  -- 1.78% return
(1, '2023-08-31', -0.0201), -- -2.01% return
(1, '2023-09-30', 0.0143),  -- 1.43% return
(1, '2023-10-31', 0.0267),  -- 2.67% return
(1, '2023-11-30', 0.0378),  -- 3.78% return
(1, '2023-12-31', 0.0312);  -- 3.12% return
```

---

## 🔗 Entity Relationship Diagram

```
┌─────────────────┐
│   portfolios    │
├─────────────────┤
│ id (PK)         │
│ portfolio_name  │
│ description     │
│ created_at      │
└────────┬────────┘
         │
         │ 1:N
         │
         ▼
┌─────────────────────┐
│ portfolio_returns   │
├─────────────────────┤
│ id (PK)             │
│ portfolio_id (FK)   │
│ month_end_date      │
│ portfolio_return    │
│ created_at          │
└─────────────────────┘

┌─────────────────┐
│   benchmarks    │
├─────────────────┤
│ id (PK)         │
│ benchmark_name  │
│ description     │
│ created_at      │
└────────┬────────┘
         │
         │ 1:N
         │
         ▼
┌─────────────────────┐
│ benchmark_returns   │
├─────────────────────┤
│ id (PK)             │
│ benchmark_id (FK)   │
│ month_end_date      │
│ benchmark_return    │
│ created_at          │
└─────────────────────┘
```

---

## 🚀 Quick Setup Script

Run this complete SQL script in your Supabase SQL Editor:

```sql
-- Create portfolios table
CREATE TABLE portfolios (
    id SERIAL PRIMARY KEY,
    portfolio_name TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create portfolio_returns table
CREATE TABLE portfolio_returns (
    id SERIAL PRIMARY KEY,
    portfolio_id INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    month_end_date DATE NOT NULL,
    portfolio_return NUMERIC NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(portfolio_id, month_end_date)
);

-- Create benchmarks table
CREATE TABLE benchmarks (
    id SERIAL PRIMARY KEY,
    benchmark_name TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create benchmark_returns table
CREATE TABLE benchmark_returns (
    id SERIAL PRIMARY KEY,
    benchmark_id INTEGER NOT NULL REFERENCES benchmarks(id) ON DELETE CASCADE,
    month_end_date DATE NOT NULL,
    benchmark_return NUMERIC NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(benchmark_id, month_end_date)
);

-- Create indexes for better performance
CREATE INDEX idx_portfolio_returns_portfolio_id ON portfolio_returns(portfolio_id);
CREATE INDEX idx_portfolio_returns_date ON portfolio_returns(month_end_date);
CREATE INDEX idx_benchmark_returns_benchmark_id ON benchmark_returns(benchmark_id);
CREATE INDEX idx_benchmark_returns_date ON benchmark_returns(month_end_date);

-- Insert sample portfolios
INSERT INTO portfolios (portfolio_name, description) VALUES
('Growth Plus', 'Aggressive growth portfolio focused on high-growth equities'),
('Global Dividend', 'Income-focused portfolio with global dividend stocks'),
('Secure Income', 'Conservative fixed-income portfolio'),
('Global Macro Opportunities', 'Tactical allocation based on macro trends');

-- Insert sample benchmarks
INSERT INTO benchmarks (benchmark_name, description) VALUES
('S&P 500', 'Standard & Poor''s 500 Index'),
('MSCI World', 'MSCI World Index - Global equity benchmark'),
('Secure Income Benchmark', 'Custom fixed-income benchmark');

-- Insert sample portfolio returns (Growth Plus - 2023)
INSERT INTO portfolio_returns (portfolio_id, month_end_date, portfolio_return) VALUES
(1, '2023-01-31', 0.0234),
(1, '2023-02-28', -0.0156),
(1, '2023-03-31', 0.0412),
(1, '2023-04-30', 0.0189),
(1, '2023-05-31', 0.0267),
(1, '2023-06-30', 0.0345),
(1, '2023-07-31', 0.0198),
(1, '2023-08-31', -0.0223),
(1, '2023-09-30', 0.0156),
(1, '2023-10-31', 0.0289),
(1, '2023-11-30', 0.0401),
(1, '2023-12-31', 0.0334);

-- Insert sample benchmark returns (S&P 500 - 2023)
INSERT INTO benchmark_returns (benchmark_id, month_end_date, benchmark_return) VALUES
(1, '2023-01-31', 0.0212),
(1, '2023-02-28', -0.0134),
(1, '2023-03-31', 0.0389),
(1, '2023-04-30', 0.0167),
(1, '2023-05-31', 0.0245),
(1, '2023-06-30', 0.0323),
(1, '2023-07-31', 0.0178),
(1, '2023-08-31', -0.0201),
(1, '2023-09-30', 0.0143),
(1, '2023-10-31', 0.0267),
(1, '2023-11-30', 0.0378),
(1, '2023-12-31', 0.0312);
```

---

## 📝 Data Requirements

### Return Data Format
- Returns should be stored as **decimal values** (not percentages)
- Example: 5% return = 0.05
- Negative returns are allowed (e.g., -2.5% = -0.025)

### Date Format
- Use **YYYY-MM-DD** format for dates
- Typically month-end dates (last day of each month)
- Example: '2023-01-31', '2023-02-28', etc.

### Data Frequency
- The application expects **monthly** return data
- Ensure consistent date intervals for accurate analytics

---

## 🔐 Row Level Security (RLS)

For production use, consider enabling RLS policies:

```sql
-- Enable RLS
ALTER TABLE portfolios ENABLE ROW LEVEL SECURITY;
ALTER TABLE portfolio_returns ENABLE ROW LEVEL SECURITY;
ALTER TABLE benchmarks ENABLE ROW LEVEL SECURITY;
ALTER TABLE benchmark_returns ENABLE ROW LEVEL SECURITY;

-- Example: Allow read access to authenticated users
CREATE POLICY "Allow read access to authenticated users" ON portfolios
    FOR SELECT TO authenticated USING (true);

CREATE POLICY "Allow read access to authenticated users" ON portfolio_returns
    FOR SELECT TO authenticated USING (true);

CREATE POLICY "Allow read access to authenticated users" ON benchmarks
    FOR SELECT TO authenticated USING (true);

CREATE POLICY "Allow read access to authenticated users" ON benchmark_returns
    FOR SELECT TO authenticated USING (true);
```

---

## ✅ Verification Queries

After setup, verify your data:

```sql
-- Check portfolios
SELECT * FROM portfolios;

-- Check portfolio returns count
SELECT p.portfolio_name, COUNT(pr.id) as return_count
FROM portfolios p
LEFT JOIN portfolio_returns pr ON p.id = pr.portfolio_id
GROUP BY p.portfolio_name;

-- Check benchmarks
SELECT * FROM benchmarks;

-- Check benchmark returns count
SELECT b.benchmark_name, COUNT(br.id) as return_count
FROM benchmarks b
LEFT JOIN benchmark_returns br ON b.id = br.benchmark_id
GROUP BY b.benchmark_name;

-- Check date ranges
SELECT 
    MIN(month_end_date) as earliest_date,
    MAX(month_end_date) as latest_date
FROM portfolio_returns;
```

---

## 🎯 Next Steps

1. **Create Tables:** Run the Quick Setup Script in Supabase SQL Editor
2. **Verify Data:** Run verification queries to ensure data is loaded
3. **Update .env:** Ensure SUPABASE_URL and SUPABASE_KEY are correct
4. **Restart Services:** Run `stop_all_services.bat` then `start_all_services.bat`
5. **Test Application:** Try queries like "Calculate volatility for Growth Plus from 2023-01-31 to 2023-12-31"

---

## 📚 Additional Resources

- **Supabase Documentation:** https://supabase.com/docs
- **SQL Editor:** Available in your Supabase dashboard
- **Table Editor:** Visual interface for managing data
- **API Documentation:** Auto-generated REST API for your tables

---

## 🆘 Troubleshooting

### Issue: "Portfolio not found"
- Verify portfolio_name matches exactly (case-sensitive)
- Check: `SELECT * FROM portfolios;`

### Issue: "No returns data"
- Verify portfolio_returns table has data
- Check date range matches your data
- Query: `SELECT * FROM portfolio_returns WHERE portfolio_id = 1;`

### Issue: "Connection error"
- Verify SUPABASE_URL and SUPABASE_KEY in .env
- Check Supabase project is active
- Verify network connectivity

---

**Database schema documentation complete!** 🎉
