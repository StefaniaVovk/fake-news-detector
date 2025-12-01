# Fake News Detector

A containerized demo that trains and serves a transformer-based fake-news classifier. The stack is composed of:

- **frontend** – React SPA for training and inference
- **backend** – ASP.NET Core API that proxies requests to the ML service
- **ml-service** – FastAPI application wrapping Hugging Face models

## How to Launch

This project is designed to be run with Docker. Follow these steps to get the application up and running.

### 1. Prerequisites

- **Docker Desktop:** Ensure you have Docker Desktop installed and running, with support for Compose v2.

### 2. Clone the Repository

Clone this repository to your local machine:

```bash
git clone <repository-url>
cd <repository-directory>
```

### 3. Build and Run the Application

From the root of the repository, run the following command to build the Docker images and start the services:

```bash
docker compose up --build -d
```

The `-d` flag runs the containers in detached mode.

### 4. Access the Application

Once the containers are running, you can access the services at the following URLs:

| Service    | URL                     |
|------------|-------------------------|
| Frontend   | http://localhost:3000   |
| Backend    | http://localhost:5000   |
| ML Service | http://localhost:8000   |

## Development Workflow

If you prefer to run the services without Docker, follow these steps:

### Backend

```bash
# From the backend directory
cd backend
dotnet run --project src/RealFakeNews.csproj
```

### ML Service

```bash
# From the ml-service directory
cd ml-service
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
# From the frontend/app directory
cd frontend/app
npm install
npm start
```

Remember to update `REACT_APP_API_URL` in `frontend/app/.env` if your backend is not running on `http://localhost:5000`.

## Project Structure

```
.
├── backend/
│   ├── src/              # ASP.NET Core source code
│   └── Dockerfile
├── frontend/
│   ├── app/              # React application source code
│   └── Dockerfile
├── ml-service/           # FastAPI and Hugging Face ML service
└── docker-compose.yml
└── fakenewsdb.sql        # A file with a database structure that can be imported
```

## Known Limitations

- Training large models on a CPU can be slow. A GPU is recommended but not required.
- The React application has basic styling. Feel free to replace it with a design system of your choice.
- Backend proxy wired via `ML_BASE_URL`
- Docker Compose with healthchecks
- Tests & pinned dependencies
