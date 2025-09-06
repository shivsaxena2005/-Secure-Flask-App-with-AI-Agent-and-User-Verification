-- Create database
CREATE DATABASE mydb;

-- Switch to database (on PostgreSQL shell use: \c mydb)
-- In scripts, you don’t use "USE mydb;"

-- Create table
CREATE TABLE first (
    sno SERIAL PRIMARY KEY,         -- auto-increment integer in Postgres
    name VARCHAR(20) NOT NULL,
    password VARCHAR(255) NOT NULL,
    email VARCHAR(500) UNIQUE NOT NULL,
    branch VARCHAR(10) NOT NULL,
    address VARCHAR(20) NOT NULL
);
