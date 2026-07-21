import sys, asyncio
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.database.database import AsyncSessionLocal
from app.database.models import AssessmentAssignment
from sqlalchemy import select
from sqlalchemy.orm import selectinload

async def dump_questions():
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(AssessmentAssignment).options(selectinload(AssessmentAssignment.assessment))
        )
        assignments = result.scalars().all()
        print(f"Total assignments in DB: {len(assignments)}")
        for asgn in assignments:
            asm = asgn.assessment
            if not asm:
                continue
            print(f"\n--- Assessment ID: {asm.id} | Name: {asm.name} ---")
            for idx, q in enumerate(asm.questions or []):
                print(f"  Q{idx+1}: type={q.get('type')!r}, subject={q.get('subject')!r}, topic={q.get('topic')!r}, language={q.get('language')!r}, has_options={bool(q.get('options'))}")

if __name__ == '__main__':
    asyncio.run(dump_questions())
