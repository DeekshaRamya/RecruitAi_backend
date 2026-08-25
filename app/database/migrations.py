import asyncio
import logging
import sys
from sqlalchemy import text, inspect
from app.database.database import engine

logger = logging.getLogger("recruitai-backend")

async def check_postgresql_locks(conn):
    """
    Checks for active locks or blocking transactions on target tables in PostgreSQL.
    """
    try:
        lock_query = text("""
            SELECT 
                a.pid, 
                a.usename, 
                a.state, 
                a.query, 
                l.mode, 
                l.granted
            FROM pg_locks l
            JOIN pg_stat_activity a ON l.pid = a.pid
            WHERE l.relation::regclass::text IN ('candidate_answers', 'assessment_results', 'assessments', 'assessment_assignments')
              AND a.pid != pg_backend_pid();
        """)
        result = await conn.execute(lock_query)
        locks = result.fetchall()
        if locks:
            logger.warning(f"⚠️ Detected {len(locks)} active PostgreSQL lock(s) on target tables:")
            for lock in locks:
                logger.warning(f"   - PID {lock.pid} ({lock.usename}) [{lock.state}]: '{lock.query}' | Lock mode: {lock.mode} (Granted: {lock.granted})")
        else:
            logger.info("No blocking PostgreSQL locks detected on target tables.")
    except Exception as e:
        logger.debug(f"Lock check skipped or non-applicable: {e}")

async def terminate_blocking_locks(conn):
    """
    Detects and terminates non-system PostgreSQL sessions holding locks on target tables
    that block migration execution.
    """
    try:
        terminate_query = text("""
            SELECT pg_terminate_backend(a.pid), a.pid, a.usename, a.state, a.query
            FROM pg_locks l
            JOIN pg_stat_activity a ON l.pid = a.pid
            WHERE l.relation::regclass::text IN ('candidate_answers', 'assessment_results', 'assessments', 'assessment_assignments')
              AND a.pid != pg_backend_pid()
              AND a.state IN ('idle in transaction', 'idle in transaction (aborted)', 'active');
        """)
        result = await conn.execute(terminate_query)
        terminated = result.fetchall()
        if terminated:
            for item in terminated:
                logger.warning(f"⚠️ Terminated blocking PostgreSQL PID {item[1]} ({item[2]}) [{item[3]}]: '{item[4]}'")
        else:
            logger.info("No blocking lock sessions found to terminate.")
    except Exception as e:
        logger.debug(f"Could not terminate blocking locks: {e}")

async def check_tables_exist(conn) -> bool:
    """
    Fast query (<10ms) to check if core application tables exist.
    """
    try:
        def inspect_tables(sync_conn):
            inspector = inspect(sync_conn)
            existing = set(inspector.get_table_names())
            required = {"users", "assessments", "candidate_answers", "assessment_results", "candidate_profiles"}
            return required.issubset(existing)
        return await conn.run_sync(inspect_tables)
    except Exception as e:
        logger.debug(f"Table inspection error: {e}")
        return False

async def is_schema_up_to_date(conn) -> bool:
    """
    Fast metadata query (<15ms) to verify if database schema has all migrated columns and tables.
    Returns True if schema is up to date, False if migrations are needed.
    """
    try:
        is_postgres = conn.dialect.name == "postgresql"
        if is_postgres:
            query = text("""
                SELECT table_name, column_name 
                FROM information_schema.columns 
                WHERE (table_name = 'candidate_answers' AND column_name IN ('assessment_id', 'test_results'))
                   OR (table_name = 'assessment_results' AND column_name IN ('overall_feedback', 'warning_history'))
                   OR (table_name = 'candidate_profiles' AND column_name = 'candidate_id')
                   OR (table_name = 'assessment_recordings' AND column_name = 'cloudinary_url');
            """)
            result = await conn.execute(query)
            found_cols = set(result.fetchall())
            return len(found_cols) >= 6
        else:
            def inspect_columns(sync_conn):
                inspector = inspect(sync_conn)
                tables = inspector.get_table_names()
                if "candidate_answers" not in tables or "assessment_results" not in tables or "candidate_profiles" not in tables or "assessment_recordings" not in tables:
                    return False
                ca_cols = [c["name"] for c in inspector.get_columns("candidate_answers")]
                ar_cols = [c["name"] for c in inspector.get_columns("assessment_results")]
                return "assessment_id" in ca_cols and "overall_feedback" in ar_cols
            return await conn.run_sync(inspect_columns)
    except Exception as e:
        logger.debug(f"Fast schema check error: {e}")
        return False

async def run_schema_migrations(target_engine):
    """
    Executes schema migrations (ALTER TABLE, UPDATE, DO blocks) safely with lock checks,
    per-statement timeouts, isolated transactions, traceback logging, and error resilience.
    """
    logger.info("Running schema migrations (ALTER TABLE)...")
    
    migrations = [
        (
            "Add admin value to userrole enum if exists",
            """
            DO $$
            BEGIN
                IF EXISTS (SELECT 1 FROM pg_type WHERE typname = 'userrole') THEN
                    BEGIN
                        ALTER TYPE userrole ADD VALUE IF NOT EXISTS 'admin';
                    EXCEPTION WHEN duplicate_object THEN NULL;
                    END;
                    BEGIN
                        ALTER TYPE userrole ADD VALUE IF NOT EXISTS 'ADMIN';
                    EXCEPTION WHEN duplicate_object THEN NULL;
                    END;
                END IF;
            END $$;
            """
        ),
        (
            "Add assessment_id column to candidate_answers",
            "ALTER TABLE candidate_answers ADD COLUMN IF NOT EXISTS assessment_id UUID;"
        ),
        (
            "Alter question_id column type to VARCHAR(4000) in candidate_answers",
            "ALTER TABLE candidate_answers ALTER COLUMN question_id TYPE VARCHAR(4000);"
        ),
        (
            "Backfill assessment_id in candidate_answers from assessment_assignments",
            """
            UPDATE candidate_answers ca
            SET assessment_id = aa.assessment_id
            FROM assessment_assignments aa
            WHERE ca.assignment_id = aa.id AND ca.assessment_id IS NULL;
            """
        ),
        (
            "Add foreign key constraint fk_candidate_answers_assessment",
            """
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints WHERE constraint_name = 'fk_candidate_answers_assessment') THEN
                    ALTER TABLE candidate_answers ADD CONSTRAINT fk_candidate_answers_assessment FOREIGN KEY (assessment_id) REFERENCES assessments(id) ON DELETE CASCADE;
                END IF;
            END $$;
            """
        ),
        (
            "Add overall_feedback column to assessment_results",
            "ALTER TABLE assessment_results ADD COLUMN IF NOT EXISTS overall_feedback VARCHAR(4000);"
        ),
        (
            "Add overall_strengths column to assessment_results",
            "ALTER TABLE assessment_results ADD COLUMN IF NOT EXISTS overall_strengths VARCHAR(4000);"
        ),
        (
            "Add overall_weaknesses column to assessment_results",
            "ALTER TABLE assessment_results ADD COLUMN IF NOT EXISTS overall_weaknesses VARCHAR(4000);"
        ),
        (
            "Add hiring_recommendation column to assessment_results",
            "ALTER TABLE assessment_results ADD COLUMN IF NOT EXISTS hiring_recommendation VARCHAR(255);"
        ),
        (
            "Add auto_submitted column to assessment_results",
            "ALTER TABLE assessment_results ADD COLUMN IF NOT EXISTS auto_submitted BOOLEAN DEFAULT FALSE;"
        ),
        (
            "Add submission_reason column to assessment_results",
            "ALTER TABLE assessment_results ADD COLUMN IF NOT EXISTS submission_reason VARCHAR(1000);"
        ),
        (
            "Add warning_count column to assessment_results",
            "ALTER TABLE assessment_results ADD COLUMN IF NOT EXISTS warning_count INTEGER DEFAULT 0;"
        ),
        (
            "Add warning_history column to assessment_results",
            "ALTER TABLE assessment_results ADD COLUMN IF NOT EXISTS warning_history JSON;"
        ),
        (
            "Add passed_test_cases column to candidate_answers",
            "ALTER TABLE candidate_answers ADD COLUMN IF NOT EXISTS passed_test_cases INTEGER;"
        ),
        (
            "Add failed_test_cases column to candidate_answers",
            "ALTER TABLE candidate_answers ADD COLUMN IF NOT EXISTS failed_test_cases INTEGER;"
        ),
        (
            "Add run_time column to candidate_answers",
            "ALTER TABLE candidate_answers ADD COLUMN IF NOT EXISTS run_time DOUBLE PRECISION;"
        ),
        (
            "Add code_output column to candidate_answers",
            "ALTER TABLE candidate_answers ADD COLUMN IF NOT EXISTS code_output VARCHAR(4000);"
        ),
        (
            "Add test_results column to candidate_answers",
            "ALTER TABLE candidate_answers ADD COLUMN IF NOT EXISTS test_results JSON;"
        ),
        (
            "Create candidate_groups table",
            """
            CREATE TABLE IF NOT EXISTS candidate_groups (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                name VARCHAR(255) NOT NULL,
                description VARCHAR(1000),
                created_by UUID REFERENCES users(id) ON DELETE SET NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
            );
            """
        ),
        (
            "Create candidate_group_members table",
            """
            CREATE TABLE IF NOT EXISTS candidate_group_members (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                group_id UUID NOT NULL REFERENCES candidate_groups(id) ON DELETE CASCADE,
                candidate_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                added_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
                CONSTRAINT uq_group_candidate UNIQUE (group_id, candidate_id)
            );
            """
        ),
        (
            "Add created_by column to assessments",
            "ALTER TABLE assessments ADD COLUMN IF NOT EXISTS created_by UUID REFERENCES users(id) ON DELETE SET NULL;"
        ),
        (
            "Create ai_usage_logs table",
            """
            CREATE TABLE IF NOT EXISTS ai_usage_logs (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID REFERENCES users(id) ON DELETE SET NULL,
                user_name VARCHAR(255),
                role VARCHAR(50),
                ai_provider VARCHAR(100) NOT NULL,
                model_name VARCHAR(100) NOT NULL,
                feature_name VARCHAR(150) NOT NULL,
                request_id VARCHAR(255),
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                request_time TIMESTAMP WITH TIME ZONE NOT NULL,
                response_time_ms INTEGER NOT NULL DEFAULT 0,
                status VARCHAR(50) NOT NULL DEFAULT 'Success',
                error_message TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
            );
            """
        ),
        (
            "Create candidate_profiles table",
            """
            CREATE TABLE IF NOT EXISTS candidate_profiles (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                candidate_id UUID NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
                resume_filename VARCHAR(255),
                resume_score INTEGER,
                python_score INTEGER,
                sql_score INTEGER,
                aptitude_score INTEGER,
                english_score INTEGER,
                resume_analysis JSON,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
            );
            """
        ),
        (
            "Migrate legacy candidate columns from users to candidate_profiles if present",
            """
            DO $$
            BEGIN
                IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'resume_filename') THEN
                    INSERT INTO candidate_profiles (id, candidate_id, resume_filename, resume_score, python_score, sql_score, aptitude_score, english_score, resume_analysis)
                    SELECT gen_random_uuid(), id, resume_filename, resume_score, python_score, sql_score, aptitude_score, english_score, resume_analysis
                    FROM users
                    WHERE resume_filename IS NOT NULL OR resume_score IS NOT NULL OR python_score IS NOT NULL OR sql_score IS NOT NULL OR aptitude_score IS NOT NULL OR english_score IS NOT NULL OR resume_analysis IS NOT NULL
                    ON CONFLICT (candidate_id) DO NOTHING;

                    ALTER TABLE users DROP COLUMN IF EXISTS resume_filename;
                    ALTER TABLE users DROP COLUMN IF EXISTS resume_score;
                    ALTER TABLE users DROP COLUMN IF EXISTS python_score;
                    ALTER TABLE users DROP COLUMN IF EXISTS sql_score;
                    ALTER TABLE users DROP COLUMN IF EXISTS aptitude_score;
                    ALTER TABLE users DROP COLUMN IF EXISTS english_score;
                    ALTER TABLE users DROP COLUMN IF EXISTS resume_analysis;
                END IF;
            END $$;
            """
        ),
        (
            "Create assessment_recordings table if not exists",
            """
            CREATE TABLE IF NOT EXISTS assessment_recordings (
                id UUID PRIMARY KEY,
                assignment_id UUID REFERENCES assessment_assignments(id) ON DELETE CASCADE,
                assessment_id UUID NOT NULL REFERENCES assessments(id) ON DELETE CASCADE,
                candidate_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
                ended_at TIMESTAMP WITH TIME ZONE,
                duration INTEGER,
                status VARCHAR(50) DEFAULT 'INITIALIZED' NOT NULL,
                storage_provider VARCHAR(50) DEFAULT 'cloudinary' NOT NULL,
                cloudinary_public_id VARCHAR(255),
                cloudinary_url VARCHAR(1000),
                mime_type VARCHAR(100),
                file_size BIGINT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL
            );
            CREATE INDEX IF NOT EXISTS ix_assessment_recordings_assignment_id ON assessment_recordings (assignment_id);
            CREATE INDEX IF NOT EXISTS ix_assessment_recordings_assessment_id ON assessment_recordings (assessment_id);
            CREATE INDEX IF NOT EXISTS ix_assessment_recordings_candidate_id ON assessment_recordings (candidate_id);
            """
        ),
        (
            "Add missing columns to assessment_recordings if existing",
            """
            DO $$
            BEGIN
                IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'assessment_recordings') THEN
                    ALTER TABLE assessment_recordings ADD COLUMN IF NOT EXISTS started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP;
                    ALTER TABLE assessment_recordings ADD COLUMN IF NOT EXISTS ended_at TIMESTAMP WITH TIME ZONE;
                    ALTER TABLE assessment_recordings ADD COLUMN IF NOT EXISTS mime_type VARCHAR(100);
                    ALTER TABLE assessment_recordings ADD COLUMN IF NOT EXISTS cloudinary_url VARCHAR(1000);
                    ALTER TABLE assessment_recordings ADD COLUMN IF NOT EXISTS video_url VARCHAR(1000);
                    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'assessment_recordings' AND column_name = 'video_url') THEN
                        UPDATE assessment_recordings SET cloudinary_url = COALESCE(cloudinary_url, video_url) WHERE cloudinary_url IS NULL;
                    END IF;
                END IF;
            END $$;
            """
        ),
        (
            "Create assessment_recording_chunks table if not exists",
            """
            CREATE TABLE IF NOT EXISTS assessment_recording_chunks (
                id UUID PRIMARY KEY,
                recording_id UUID NOT NULL REFERENCES assessment_recordings(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                storage_reference VARCHAR(1000),
                file_size BIGINT,
                uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
                status VARCHAR(50) DEFAULT 'COMPLETED' NOT NULL
            );
            CREATE INDEX IF NOT EXISTS ix_assessment_recording_chunks_recording_id ON assessment_recording_chunks (recording_id);
            """
        ),
    ]

    is_postgres = target_engine.dialect.name == "postgresql"

    # Check for active PostgreSQL locks before starting migrations
    if is_postgres:
        try:
            async with target_engine.connect() as check_conn:
                await check_postgresql_locks(check_conn)
        except Exception as lock_err:
            logger.warning(f"Could not perform pre-migration lock check: {lock_err}")

    success_count = 0
    failure_count = 0
    total_steps = len(migrations)

    for step, (desc, sql_stmt) in enumerate(migrations, 1):
        logger.info(f"Migration [{step}/{total_steps}] Starting: '{desc}'...")
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                # Execute each migration statement in an isolated transaction block
                async with target_engine.begin() as conn:
                    if is_postgres:
                        await conn.execute(text("SET LOCAL lock_timeout = '5s';"))
                        await conn.execute(text("SET LOCAL statement_timeout = '10s';"))
                    
                    await asyncio.wait_for(conn.execute(text(sql_stmt)), timeout=10.0)
                    
                logger.info(f"Migration [{step}/{total_steps}] Completed successfully: '{desc}'.")
                success_count += 1
                break
            except Exception as exc:
                is_lock_error = "LockNotAvailable" in str(exc) or "lock timeout" in str(exc).lower()
                if attempt < max_retries and is_postgres and is_lock_error:
                    logger.warning(
                        f"⚠️ Migration [{step}/{total_steps}] Attempt {attempt}/{max_retries} hit lock timeout for '{desc}'. "
                        "Attempting to terminate blocking database sessions and retrying..."
                    )
                    try:
                        async with target_engine.connect() as term_conn:
                            await terminate_blocking_locks(term_conn)
                    except Exception as t_err:
                        logger.debug(f"Failed to clear blocking sessions: {t_err}")
                    await asyncio.sleep(1.0)
                else:
                    failure_count += 1
                    logger.error(
                        f"🚨 Migration [{step}/{total_steps}] Failed for '{desc}': {exc}",
                        exc_info=True
                    )
                    logger.warning(f"Continuing with remaining migrations despite failure in step {step}.")
                    break

    if failure_count == 0:
        logger.info("Database migrations completed successfully.")
    else:
        logger.warning(f"Database migrations completed with warnings: {success_count}/{total_steps} succeeded, {failure_count} failed.")

async def main_cli():
    """
    CLI runner for database migrations.
    Usage: python -m app.database.migrations OR python migrate.py
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger.info("Starting standalone database migration runner...")
    await run_schema_migrations(engine)
    logger.info("Migration runner finished.")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main_cli())
