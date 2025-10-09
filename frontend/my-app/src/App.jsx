import React, { useState} from "react";
import Visualization from "./Visualization";
import ShapHeatmap from "./ShapHeatmap";
import IgHeatmap from "./IgHeatmap";

export default function App() {
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
      const res = await fetch(`${process.env.REACT_APP_API_URL}/api/ml/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ test_size: testSize, model_name: selectedModel}),
      });
      if (!res.ok) throw new Error("Не вдалося запустити тренування");

      setModelTrained(false);
      setMetrics(null);

      const pollStatus = async () => {
        try {
          const statusRes = await fetch(`${process.env.REACT_APP_API_URL}/api/ml/analyze/status?model_name=${selectedModel}`);
          const statusData = await statusRes.json();

          if (!statusData.running) {
            setModelTrained(true);
            setMetrics(statusData.metrics || {});

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
    if (selectedModel === "logreg" && (method === "ig" || method === "tcav")) {
      alert("⚠️ Цей тип пояснення недоступний для Logistic Regression. Використай BERT-tiny.");
      return;
    }

    try {
      const res = await fetch(`${process.env.REACT_APP_API_URL}/api/ml/interpret/${method.toUpperCase()}?news_id=${selectedNewsId}&model_name=${selectedModel}`, {
        method: "POST"
      });
      const data = await res.json();
      setExplanations((prev) => ({ ...prev, [method.toUpperCase()]: data.payload }));
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
    <div className="flex h-screen">
      {/* Ліва панель */}
      <div className="w-1/2 p-6 border-r border-gray-300 overflow-y-auto">
        <h1 className="text-2xl font-bold mb-4">Fake News Detection</h1>

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
            className="mt-2 px-4 py-2 bg-blue-500 text-white rounded"
            onClick={handleUpload}
          >
            Відправити у бекенд
          </button>
        </div>

        {/* Навчання моделі */}
        <div className="mb-6">
          <h2 className="text-lg font-semibold">ML Модель</h2>

          <div className="mb-2">
            <label className="mr-2">Модель:</label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="border p-1 rounded"
            >
              <option value="logreg">Logistic Regression (через BERT-ембедінги)</option>
              <option value="bert-tiny">BERT-tiny fine-tuned (mrm8488)</option>
            </select>
          </div>

          <div className="mb-2">
            <label className="mr-2">Test size:</label>
            <select
              value={testSize}
              onChange={(e) => setTestSize(parseFloat(e.target.value))}
              className="border p-1 rounded"
            >
              <option value={0.2}>20%</option>
              <option value={0.25}>25%</option>
              <option value={0.3}>30%</option>
              <option value={0.4}>40%</option>
            </select>
          </div>
          <button
            className="px-4 py-2 bg-green-500 text-white rounded mr-2"
            onClick={handleAnalyze}
          >
            Навчити модель
          </button>

          <button
            className="px-4 py-2 bg-purple-500 text-white rounded"
            onClick={handleRandomPredict}
            disabled={!modelTrained}
          >
            Рандомний прогноз
          </button>
        </div>

        {/* Метрики */}
        {metrics && (
          <div className="mb-6 p-4 bg-gray-100 border rounded">
            <h3 className="font-semibold">📊 Метрики моделі</h3>
            <ul>
              <li>Accuracy: {Number(metrics.accuracy).toFixed(3)}</li>
              <li>Precision: {Number(metrics.precision).toFixed(3)}</li>
              <li>Recall: {Number(metrics.recall).toFixed(3)}</li>
              <li>F1-score: {Number(metrics.f1).toFixed(3)}</li>
            </ul>
          </div>
        )}

        {/* Рандомний прогноз */}
        {randomResult && (
          <div className="mb-6 p-4 bg-gray-100 border rounded">
            <h3 className="font-semibold">🎲 Рандомний прогноз</h3>
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
            className="border w-full p-2 rounded"
            placeholder="Введіть текст новини..."
            value={newsText}
            onChange={(e) => setNewsText(e.target.value)}
          />
          <button
            className="mt-2 px-4 py-2 bg-indigo-500 text-white rounded"
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
      <div className="w-1/2 p-6 overflow-y-auto">
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
          <div className="flex gap-2 mt-2">
            <button className="px-3 py-1 bg-yellow-500 text-white rounded" onClick={() => fetchExplanation("shap")}>
              SHAP
            </button>
            <button className="px-3 py-1 bg-red-500 text-white rounded" onClick={() => fetchExplanation("ig")}>
              IG
            </button>
            <button className="px-3 py-1 bg-gray-700 text-white rounded" disabled>
              TCAV
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
                  <ShapHeatmap payload={data} text={randomResult.text} />
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
  );
}
