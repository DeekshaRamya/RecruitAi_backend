-- PostgreSQL Schema DDL Script for RecruitAI
-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY,
    full_name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password VARCHAR(255), -- Nullable for recruiters utilizing Microsoft OAuth
    phone VARCHAR(50),
    role VARCHAR(50) NOT NULL, -- 'candidate' or 'recruiter'
    microsoft_id VARCHAR(255) UNIQUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- Create Indexes for fast user lookups
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_microsoft_id ON users(microsoft_id);

-- Create Login History table
CREATE TABLE IF NOT EXISTS login_history (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    login_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    ip_address VARCHAR(45),
    device VARCHAR(255),
    CONSTRAINT fk_login_history_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Create index on user_id in login_history for faster metrics/history tracking
CREATE INDEX IF NOT EXISTS idx_login_history_user_id ON login_history(user_id);

-- Create Candidate Profiles table (stores candidate resume & assessment score feature data)
CREATE TABLE IF NOT EXISTS candidate_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    candidate_id UUID NOT NULL UNIQUE,
    resume_filename VARCHAR(255),
    resume_score INTEGER,
    python_score INTEGER,
    sql_score INTEGER,
    aptitude_score INTEGER,
    english_score INTEGER,
    resume_analysis JSON,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    CONSTRAINT fk_candidate_profiles_user FOREIGN KEY (candidate_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Create index on candidate_id in candidate_profiles
CREATE INDEX IF NOT EXISTS idx_candidate_profiles_candidate_id ON candidate_profiles(candidate_id);

