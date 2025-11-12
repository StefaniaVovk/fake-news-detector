import React, { useState, useEffect } from "react";
import Visualization from "./Visualization";
import ShapHeatmap from "./ShapHeatmap";
import IgHeatmap from "./IgHeatmap";
import Plots from "./Plots";
import { useNavigate, useLocation} from "react-router-dom";

export default function App({ setUser }) {
  const [files, setFiles] = useState([]);
  const [newsText, setNewsText] = useState("");
  const [output, setOutput] = useState(null);
  const [modelTrained, setModelTrained] = useState(false);
  const [metrics, setMetrics] = useState(null);
  const [randomResult, setRandomResult] = useState(null);
  const [explanations, setExplanations] = useState({});
  const [plotDataUMAP, setPlotDataUMAP] = useState(null);
  const [plotDataTSNE, setPlotDataTSNE] = useState(null);
  const [selectedModel, setSelectedModel] = useState("logreg");
  const [testSize, setTestSize] = useState(0.3);
  const [selectedNewsId, setSelectedNewsId] = useState(null);
  const [plotsReady, setPlotsReady] = useState(false);
  
  const location = useLocation();

  const callApi = async (url, method = "GET", body = null, setFunc = setOutput) => {
    const opts = { method, headers: { "Content-Type": "application/json" } };
    if (body) opts.body = JSON.stringify(body);
    try {
      const res = await fetch(`${process.env.REACT_APP_API_URL}${url}`, opts);
      const data = await res.json();
      setFunc(data);
    } catch (err) {
      console.error("API error:", err);
      setFunc({ error: "Помилка підключення до API" });
    }
  };

  // --- Завантаження файлів ---
  const handleFileUpload = (event) => {
    setFiles(Array.from(event.target.files));
  };

  const navigate = useNavigate();
  const user = JSON.parse(localStorage.getItem("user"));

  const handleLogout = () => {
    localStorage.removeItem("user");
    setUser(null);
    navigate("/");
  };

  const handleUpload = async () => {
    if (!files.length) {
      alert("Будь ласка, виберіть файли для завантаження");
      return;
    }
    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));

    try {
      const res = await fetch(`${process.env.REACT_APP_API_URL}/api/ml/preprocess`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Помилка сервера");

      const data = await res.json();
      console.log("✅ Файли надіслані:", data);
      alert("Файли успішно надіслані!");
    } catch (err) {
      console.error("Upload error:", err);
      alert("❌ Помилка відправки файлів");
    }
  };

  // --- Тренування моделі ---
  const handleAnalyze = async () => {
    try {
      // Додаємо параметри для AnalyzeRequest
      const currentModelParams = {
          test_size: testSize,
          model_name: selectedModel,
          max_iter: 1000, // Дефолтні значення для logreg
          C: 5.0,
          solver: "liblinear"
      };

      const res = await fetch(`${process.env.REACT_APP_API_URL}/api/ml/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(currentModelParams), // Надсилаємо всі параметри
      });
      if (!res.ok) throw new Error("Не вдалося запустити тренування");

      setModelTrained(false);
      setMetrics(null);
      setPlotsReady(false);

      const pollStatus = async () => {
        try {
          const statusRes = await fetch(`${process.env.REACT_APP_API_URL}/api/ml/analyze/status?model_name=${selectedModel}`);
          const statusData = await statusRes.json();

          if (!statusData.running && statusData.ready) {
            setModelTrained(true);
            setMetrics(statusData.metrics || {});
            setPlotsReady(true);

            await fetchVisualization("UMAP", setPlotDataUMAP);
            await fetchVisualization("TSNE", setPlotDataTSNE);
            console.log("✅ Тренування завершено", statusData.metrics);
          } else {
            setTimeout(pollStatus, 2000);
          }
        } catch (err) {
          console.error("Помилка при отриманні статусу:", err);
        }
      };

      pollStatus();
    } catch (err) {
      console.error("Analyze error:", err);
      alert("Помилка при тренуванні моделі");
    }
  };

  useEffect(() => {
    if (location.state?.retrainStarted) {
      console.log("🔁 Автоматичне оновлення після перенавчання...");
      handleAnalyze();
      navigate(location.pathname, { replace: true, state: {} });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [location.state]);


  // --- Прогноз ---
  const handleRandomPredict = async () => {
    try {
      const res = await fetch(`${process.env.REACT_APP_API_URL}/api/ml/random_predict?model_name=${selectedModel}`);
      const data = await res.json();
      setRandomResult(data);
      setSelectedNewsId(data.news_id);

    } catch (err) {
      console.error("Random predict error:", err);
      alert("Помилка при отриманні прогнозу");
    }
  };

  // --- Пояснення ---
  const fetchExplanation = async (method) => {
    if (!selectedNewsId) {
      alert("⚠️ Спочатку отримай рандомну новину, щоб створити пояснення!");
      return;
    }

    // Перевіряємо, чи метод підтримується обраною моделлю
    if (selectedModel === "logreg" && (method === "ig" || method === "lime")) {
      alert("⚠️ Цей тип пояснення недоступний для Logistic Regression. Використай BERT-tiny.");
      return;
    }

    try {
      const res = await fetch(`${process.env.REACT_APP_API_URL}/api/ml/interpret/${method.toUpperCase()}?news_id=${selectedNewsId}&model_name=${selectedModel}`, {
        method: "POST"
      });
      const data = await res.json();
      setExplanations((prev) => ({
        ...prev, [method.toUpperCase()]: data,
      }));
      alert(`✅ Пояснення ${method.toUpperCase()} отримано! Fidelity: ${data.fidelity}`);
    } catch (err) {
      console.error("Interpretation error:", err);
      alert("⚠️ Сталася помилка при отриманні пояснення.");
    }
  };

  // --- Візуалізація ---
  const fetchVisualization = async (method, setData) => {
    console.log(`🔹 Fetching visualization: ${method}`);
    try {
      const res = await fetch(`${process.env.REACT_APP_API_URL}/api/ml/visualize/${method}?model_name=${selectedModel}`);
      const data = await res.json();
      console.log("📌 Visualization data:", data);

      // якщо немає точок → нічого не зберігаємо
      if (!data.points || data.points.length === 0) {
        console.warn(`⚠️ No projection points for ${method}`);
        setData(null);
        return false;
      }

      setData(data); // зберігаємо повний JSON (ids, points, labels, predicted_labels)
      return true;
    } catch (err) {
      console.error(`❌ Fetch error for ${method}:`, err);
      setData(null);
      return false;
    }
  };

  return (
    <div className="container-fluid h-100">
      <div className="row h-100">
        {/* Ліва панель */}
        <div className="col-md-6 border-end p-3 overflow-auto">
          <div className="d-flex justify-content-between align-items-center mb-4">
            <h1 className="h2 fw-bold mb-0">Fake News Detection</h1>

            <div className="d-flex align-items-center gap-3">
              <span className="text-muted">
                👤 Вітаємо, <b>{user?.name || "користувачу"}</b>
              </span>
              <button className="btn btn-outline-danger btn-sm" onClick={handleLogout}>
                Вийти
              </button>
            </div>
          </div>

          {/* Завантаження файлів */}
          <div className="mb-6">
            <h2 className="text-lg font-semibold">Завантаження файлів</h2>
            <input type="file" multiple onChange={handleFileUpload} />
            <input
              type="file"
              webkitdirectory="true"
              directory=""
              multiple
              onChange={handleFileUpload}
              style={{ display: "block", marginTop: "10px" }}
            />
            <button
              className="btn btn-outline-primary mt-2"
              onClick={handleUpload}
            >
              Обробити дані
            </button>
          </div>

          {/* Навчання моделі */}
          <div className="mb-6">
            <h2 className="text-lg font-semibold">ML Модель</h2>

            <div className="row mb-2">
              <div className="col-md-6 mb-2">
                <label className="form-label">Модель:</label>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="form-select"
                >
                  <option value="logreg">Logistic Regression (через BERT-ембедінги)</option>
                  <option value="bert-tiny">BERT-tiny fine-tuned (distilbert)</option>
                </select>
              </div>

              <div className="col-md-6 mb-2">
                <label className="form-label">Test size:</label>
                <select
                  value={testSize}
                  onChange={(e) => setTestSize(parseFloat(e.target.value))}
                  className="form-select"
                >
                  <option value={0.2}>20%</option>
                  <option value={0.25}>25%</option>
                  <option value={0.3}>30%</option>
                  <option value={0.4}>40%</option>
                </select>
              </div>
            </div>
            {/* Кнопки в рядку з відстанню */}
            <div className="d-flex gap-2">
              <button className="btn btn-outline-primary" onClick={handleAnalyze}>
                Навчити модель
              </button>

              <button
                className="btn btn-outline-primary"
                onClick={handleRandomPredict}
                disabled={!modelTrained}
              >
                Рандомний прогноз
              </button>

              <button
                className={`btn ${user?.role === "researcher" && modelTrained
                    ? "btn-outline-success"
                    : "btn-outline-secondary"
                  }`}
                disabled={user?.role !== "researcher" || !modelTrained}
                onClick={() => {
                  if (user?.role === "researcher" && modelTrained) {
                    navigate(`/improve/${selectedModel}`);
                  }
                }}
              >
                Покращити роботу моделі
              </button>
            </div>
          </div>

          {/* Метрики */}
          {metrics && (
            <div className="mt-3 p-3 bg-white rounded">
              <h3 className="fw-semibold">📊 Метрики моделі</h3>
              <ul className="mb-0">
                <li>Accuracy: {Number(metrics.accuracy).toFixed(3)}</li>
                <li>Precision: {Number(metrics.precision).toFixed(3)}</li>
                <li>Recall: {Number(metrics.recall).toFixed(3)}</li>
                <li>F1-score: {Number(metrics.f1).toFixed(3)}</li>
              </ul>
            </div>
          )}

          {/* Вивід графіків */}
          <div className="mt-6">
            <Plots model={selectedModel} plotsReady={plotsReady} />
          </div>

          {/* Рандомний прогноз */}
          {randomResult && (
            <div className="mt-4 p-3 bg-white rounded">
              <h3 className="fw-semibold">🎲 Рандомний прогноз</h3>
              <p><b>Текст новини:</b> {randomResult.text.length > 200
                ? randomResult.text.slice(0, 200) + "..."
                : randomResult.text}</p>
              <p><b>Прогноз:</b> {randomResult.prediction.predicted_label}</p>
              <p><b>Впевненість:</b> {(randomResult.prediction.probability * 100).toFixed(2)}%</p>
              <p><b>Справжня мітка:</b> {randomResult.true_label}</p>
            </div>
          )}

          {/* Введення тексту */}
          <div className="mb-6">
            <textarea
              className="form-control mb-2"
              placeholder="Введіть текст новини..."
              value={newsText}
              onChange={(e) => setNewsText(e.target.value)}
            />
            <button
              className="mt-2 btn btn-outline-primary"
              onClick={() =>
                callApi("/api/ml/predict", "POST", { news_text: newsText })
              }
              disabled={!modelTrained}
            >
              Прогноз для введеного тексту
            </button>
          </div>

          {/* Вивід */}
          {output && (
            <pre className="mt-4 p-2 bg-gray-100 border rounded">
              {JSON.stringify(output, null, 2)}
            </pre>
          )}
        </div>

        {/* Права панель */}
        <div className="col-md-6 p-3 overflow-auto">
          <h1 className="text-xl font-bold mb-4">Інтерпретація</h1>

          <h2 className="text-xl font-bold mb-4">Візуалізація</h2>

          <div className="visualizations">
            {plotDataUMAP && (
              <>
                <h3>UMAP — Справжні мітки</h3>
                <Visualization data={plotDataUMAP} labelType="label" />

                <h3>UMAP — Прогнозовані мітки</h3>
                <Visualization data={plotDataUMAP} labelType="predicted_label" />
              </>
            )}

            {plotDataTSNE && (
              <>
                <h3>t-SNE — Справжні мітки</h3>
                <Visualization data={plotDataTSNE} labelType="label" />

                <h3>t-SNE — Прогнозовані мітки</h3>
                <Visualization data={plotDataTSNE} labelType="predicted_label" />
              </>
            )}
          </div>

          {/* Кнопки пояснень */}
          <div className="mt-6">
            <h2 className="text-lg font-semibold">Пояснення</h2>
            <div className="d-flex gap-2 mt-2">
              <button className="btn btn-outline-primary" onClick={() => fetchExplanation("shap")}>
                SHAP
              </button>
              <button className="btn btn-outline-primary" onClick={() => fetchExplanation("ig")}>
                IG
              </button>
              <button className="btn btn-outline-primary" disabled>
                LIME
              </button>
            </div>
          </div>

          {/* Вивід пояснень */}
          {Object.keys(explanations).length > 0 && (
            <div className="mt-4">
              {Object.entries(explanations).map(([method, data]) => (
                <div key={method} className="mb-6">

                  {/* <h4 className="font-semibold mb-2">{method.toUpperCase()}</h4> */}

                  {/* JSON формат пояснення */}
                  {/* <pre className="bg-gray-100 p-2 rounded text-sm overflow-x-auto mb-2"> */}
                  {/* {JSON.stringify(data, null, 2)} */}
                  {/* </pre>*/}

                  {/* Візуалізація SHAP (теплова карта) */}
                  {method.toUpperCase() === "SHAP" && randomResult?.text && (
                    <ShapHeatmap payload={data.payload} text={randomResult.text} prediction={{ predicted_label: data.predicted_label }} />
                  )}

                  {method.toUpperCase() === "IG" && randomResult?.text && (
                    <IgHeatmap payload={data} text={randomResult.text} />
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

    </div>
  );
}
