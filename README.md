# InsureSense - Customer Behavior Prediction Platform

A production-style Machine Learning web application for insurance companies to predict the likelihood of existing customers purchasing an additional vehicle insurance policy.

## Overview

InsureSense analyzes historical customer data using supervised ML classification algorithms to generate purchase probability predictions. The system helps insurance sales and marketing teams prioritize potential customers based on predicted purchase probability.

**This is a decision-support system - it does NOT:**
- Approve or reject insurance applications
- Decide insurance eligibility
- Calculate insurance premiums
- Guarantee customer purchases
- Use LLMs, RAG, LangChain, or AI Agents

## Tech Stack

### Frontend
- React.js + TypeScript
- Tailwind CSS
- Recharts (visualizations)

### Backend
- Python + FastAPI
- Pydantic + SQLAlchemy
- PostgreSQL

### Machine Learning
- Scikit-learn
- XGBoost
- Pandas + NumPy
- MLflow (experiment tracking)

### DevOps
- Docker + Docker Compose
- GitHub Actions (CI/CD)
- Pytest (testing)

## Project Structure

```
InsureSense/
├── frontend/                 # React TypeScript application
│   └── src/
│       ├── components/       # Reusable UI components
│       ├── pages/            # Page components
│       ├── services/         # API services
│       ├── hooks/            # Custom React hooks
│       └── types/            # TypeScript types
├── backend/                  # FastAPI application
│   ├── app/
│   │   ├── api/              # API routes
│   │   ├── core/             # Configuration, security
│   │   ├── schemas/          # Pydantic models
│   │   ├── models/           # SQLAlchemy models
│   │   ├── services/         # Business logic
│   │   ├── repositories/     # Data access
│   │   └── main.py           # App entry point
│   └── tests/                # Backend tests
├── ml/                       # Machine Learning pipeline
│   ├── data/
│   │   ├── raw/              # Raw dataset
│   │   └── processed/        # Processed data
│   ├── notebooks/            # Jupyter notebooks
│   ├── src/
│   │   ├── ingestion/        # Data loading
│   │   ├── validation/       # Data validation
│   │   ├── preprocessing/    # Preprocessing pipeline
│   │   ├── features/         # Feature engineering
│   │   ├── training/         # Model training
│   │   ├── evaluation/       # Model evaluation
│   │   └── inference/        # Inference utilities
│   ├── artifacts/            # Trained model artifacts
│   └── configs/              # Configuration files
├── tests/                    # Integration tests
├── docs/                     # Documentation
├── .github/workflows/        # CI/CD pipelines
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── requirements.txt
└── README.md
```

## Dataset

**Vehicle Insurance Cross-Sell Prediction** (publicly available)

- **Samples**: 50,000
- **Features**: 11 (excluding ID)
- **Target**: `Response` (binary: 0=No Purchase, 1=Purchase)
- **Class Distribution**: 87.7% / 12.3% (imbalanced ~7:1)
- **Location**: `ml/data/raw/train.csv`

### Features
| Feature | Type | Description |
|---------|------|-------------|
| Gender | Categorical | Male/Female |
| Age | Integer | 20-85 |
| Driving_License | Binary | 0/1 |
| Region_Code | Categorical | 52 regions |
| Previously_Insured | Binary | 0/1 |
| Vehicle_Age | Ordinal | <1yr / 1-2yr / >2yr |
| Vehicle_Damage | Binary | No/Yes |
| Annual_Premium | Continuous | 2,630-540,165 |
| Policy_Sales_Channel | Categorical | 126 channels |
| Vintage | Integer | 10-299 days |
| Response | Binary (Target) | 0/1 |

## Quick Start

### Prerequisites
- Docker + Docker Compose
- Node.js 18+ (for frontend development)
- Python 3.11+ (for backend development)

### Using Docker Compose (Recommended)

```bash
# Copy environment file
cp .env.example .env

# Start all services
docker-compose up -d

# Services available at:
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
# MLflow: http://localhost:5000
# PostgreSQL: localhost:5432
```

### Manual Development Setup

#### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r ../requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend
```bash
cd frontend
npm install
npm run dev
```

#### ML Pipeline
```bash
cd ml
pip install -r ../requirements.txt
jupyter lab notebooks/
```

## Development Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ Complete | Problem understanding, dataset acquisition, validation |
| 2 | ⏳ Pending | Exploratory Data Analysis |
| 3 | ⏳ Pending | ML preprocessing & feature engineering pipeline |
| 4 | ⏳ Pending | Model training, CV, hyperparameter tuning, evaluation |
| 5 | ⏳ Pending | MLflow experiment tracking & model versioning |
| 6 | ⏳ Pending | FastAPI backend & inference APIs |
| 7 | ⏳ Pending | PostgreSQL database integration |
| 8 | ⏳ Pending | React frontend & API integration |
| 9 | ⏳ Pending | Testing, Docker, GitHub Actions CI/CD |
| 10 | ⏳ Pending | Deployment, logging, monitoring, documentation |

## API Endpoints (Planned)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/auth/register` | User registration |
| POST | `/api/v1/auth/login` | User login |
| GET | `/api/v1/dashboard/summary` | Dashboard analytics |
| POST | `/api/v1/predictions` | Individual prediction |
| POST | `/api/v1/predictions/batch` | Batch prediction (CSV) |
| GET | `/api/v1/predictions` | Prediction history |
| GET | `/api/v1/models/active` | Active model info |
| GET | `/api/v1/models/metrics` | Model evaluation metrics |
| GET | `/health` | Health check |

## Modules

1. **Authentication** - JWT-based auth, roles (Admin, Employee)
2. **Analytics Dashboard** - KPIs, propensity distributions, charts
3. **Individual Prediction** - Single customer prediction form
4. **Batch Prediction** - CSV upload, bulk predictions, download
5. **Prediction History** - Filterable, paginated history
6. **Model Management** - Model version, metrics, status

## ML Algorithms (5 Required)

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. XGBoost

## Evaluation Metrics

- Accuracy (monitoring only)
- Precision, Recall, F1-Score
- ROC-AUC, PR-AUC
- Confusion Matrix
- Precision@K, Recall@K

## License

MIT License - See LICENSE file for details.

## Contributing

This is a structured development project following a 10-phase plan. Each phase requires explicit approval before proceeding.
