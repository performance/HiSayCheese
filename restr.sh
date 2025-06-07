#!/bin/bash

set -e

echo "🔄 Creating frontend directory layout..."
mkdir -p frontend/{src,public}
mkdir -p frontend/config

echo "📦 Moving frontend-specific files..."
mv next.config.* frontend/config/ 2>/dev/null || true
mv postcss.config.mjs frontend/config/ 2>/dev/null || true
mv package*.json frontend/ 2>/dev/null || true
mv tailwind.config.ts frontend/config/ 2>/dev/null || true
mv tsconfig.json frontend/ 2>/dev/null || true

echo "�� Moving frontend source code..."
mv src frontend/ 2>/dev/null || true
mv next-env.d.ts frontend/ 2>/dev/null || true

echo "🧹 Removing old empty frontend dir if it exists..."
rmdir frontend 2>/dev/null || true

echo "✅ Moving backend test artifacts..."
rm -rf tests/__pycache__

echo "✅ Finished reorganizing!"

echo -e "\n🧭 Your new structure should look like this:"
echo "
HiSayCheese/
├── backend/
│   └── ...
├── frontend/
│   ├── src/
│   ├── config/
│   ├── package.json
│   └── tsconfig.json
├── docs/
├── README.md
├── requirements.txt
└── requirements-dev.txt
"

