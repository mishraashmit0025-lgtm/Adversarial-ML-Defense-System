FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY adv_defense/ adv_defense/
# Train on synthetic flows at build time so the image serves a model out of the box.
# Mount CICIDS 2017 CSVs at /data and run `python -m adv_defense --data /data` to retrain on real traffic.
RUN python -m adv_defense --n 20000 --n-attack 300 --epochs 10
ENV MODEL_PATH=/app/results/defense.joblib
EXPOSE 8000
CMD ["uvicorn", "adv_defense.service:app", "--host", "0.0.0.0", "--port", "8000"]
