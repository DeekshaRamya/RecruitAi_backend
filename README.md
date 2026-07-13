# RecruitAI Backend

A production-ready, secure, and scalable FastAPI backend for the **RecruitAI** (AI-Powered Recruitment Assessment Platform). This backend implements clean architecture separation of concerns, robust password hashing, token rotation, and Role-Based Access Control (RBAC).

## Features

- **Candidate Auth**: Manual registration & login (using bcrypt password hashing and Pydantic validation).
- **Recruiter Auth**: Password-less authentication via Microsoft Entra ID (Azure AD) OAuth2.
- **Security**: JWT-based session security with token rotation (Access & Refresh tokens).
- **Role-Based Access Control (RBAC)**: Secure routes using custom dependencies (`require_candidate`, `require_recruiter`).
- **Structured Validation Errors**: Standardized JSON responses for request validation errors.
- **Documentation**: Automatic OpenAPI generation via Swagger UI (`/docs`) and ReDoc (`/redoc`).
- **Complete Test Suite**: Comprehensive integration test suite using an in-memory SQLite backend.

---

## Tech Stack

- **FastAPI**: Core high-performance web framework.
- **SQLAlchemy 2.0 ORM**: Python SQL toolkit and Object Relational Mapper.
- **psycopg**: Modern PostgreSQL driver.
- **PostgreSQL**: Production-grade relational database.
- **Pydantic V2**: Data parsing, serialization, and validation.
- **python-jose**: JWT signature generation & verification.
- **passlib (bcrypt)**: Secure password hashing.

---

## Getting Started

### Prerequisites

- Python 3.10+
- PostgreSQL database (or access to one)

### Installation & Local Setup

1. **Clone the Repository** (if not already done).
2. **Create and Activate a Virtual Environment**:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure Environment Variables**:
   Copy or create the `.env` file in the root directory:
   ```env
   DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/recruitai
   JWT_SECRET_KEY=your-jwt-secret-key-here
   JWT_ALGORITHM=HS256
   ACCESS_TOKEN_EXPIRE_MINUTES=60
   REFRESH_TOKEN_EXPIRE_MINUTES=10080

   # Microsoft Entra ID OAuth Configuration
   MICROSOFT_CLIENT_ID=your-microsoft-client-id-here
   MICROSOFT_CLIENT_SECRET=your-microsoft-client-secret-here
   MICROSOFT_TENANT_ID=common
   MICROSOFT_REDIRECT_URI=http://localhost:8000/auth/microsoft/callback

   FRONTEND_URL=http://localhost:5173
   ```
5. **Run the Database**:
   Ensure PostgreSQL is running locally or remotely and the database named `recruitai` is created.

6. **Start the Backend Server**:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
   *Note: Tables will be created automatically on startup.*

---

## Microsoft Entra ID (Azure AD OAuth) Setup

To configure recruiter login with Microsoft:
1. Go to the [Azure Portal](https://portal.azure.com/) and navigate to **Microsoft Entra ID** (formerly Azure Active Directory) -> **App registrations**.
2. Click **New registration**:
   - **Name**: RecruitAI Backend
   - **Supported account types**: Accounts in any organizational directory (Any Microsoft Entra ID tenant - Multitenant) and personal Microsoft accounts (e.g. Skype, Xbox).
   - **Redirect URI (optional)**: Web, and set to `http://localhost:8000/auth/microsoft/callback`.
3. After registration, copy the **Application (client) ID** and **Directory (tenant) ID** into the `.env` file (`MICROSOFT_CLIENT_ID` and `MICROSOFT_TENANT_ID`).
4. Go to **Certificates & secrets** -> **Client secrets** -> **New client secret**.
5. Copy the generated secret value into the `.env` file (`MICROSOFT_CLIENT_SECRET`).

---

## API Endpoints

### Authentication (Public)
- `POST /auth/candidate/register`: Register candidate (Email, Phone, Name, Password).
- `POST /auth/candidate/login`: Authenticate candidate.
- `GET /auth/microsoft/login`: Redirects to Microsoft Entra ID authorization.
- `GET /auth/microsoft/callback`: OAuth callback for Microsoft Entra ID (creates/logs in recruiter).
- `POST /auth/refresh`: Refresh access and refresh tokens.
- `POST /auth/logout`: Invalidates session (requires authentication).
- `GET /auth/me`: Retrieves currently authenticated user info (requires authentication).

### Candidates (Candidate Role Required)
- `GET /candidate/dashboard`: Returns candidate-specific dashboard data.
- `GET /candidate/profile`: Returns candidate profile details.

### Recruiters (Recruiter Role Required)
- `GET /recruiter/dashboard`: Returns recruiter-specific dashboard data.
- `GET /recruiter/profile`: Returns recruiter profile details.

---

## Running Tests

To run the integration test suite (uses an isolated in-memory SQLite database):
```bash
python -m pytest
```
