import React, { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";

export default function ImproveModel() {
    const { modelName } = useParams();
    const navigate = useNavigate();
    const [params, setParams] = useState({});
    const [datasetFiles, setDatasetFiles] = useState([]);
    const [feedbackStatus, setFeedbackStatus] = useState(null);

    // Приклад початкових параметрів для logreg
    useEffect(() => {
        if (modelName === "logreg") {
            setParams({
                test_size: 0.3,
                max_iter: 1000,
                C: 5.0,
                solver: "liblinear"
            });
        }
        // можна додати інші моделі
    }, [modelName]);

    const handleParamChange = (key, value) => {
        // Конвертуємо значення в числа, якщо це числові параметри
        let processedValue = value;
        if (key === "test_size" || key === "C") {
            processedValue = parseFloat(value);
            if (isNaN(processedValue)) processedValue = value; // Залишаємо як є, якщо не число
        } else if (key === "max_iter") {
            processedValue = parseInt(value, 10);
            if (isNaN(processedValue)) processedValue = value; // Залишаємо як є, якщо не число
        }
        setParams((prev) => ({ ...prev, [key]: processedValue }));
    };

    const handleDatasetFileChange = (event) => {
        setDatasetFiles(Array.from(event.target.files));
    };

    const handleSubmitFeedback = async (feedbackType) => {
        const formData = new FormData();
        formData.append("type", feedbackType);

        const payload = { model_name: modelName }; // Завжди передаємо model_name у payload
        if (feedbackType === "parameter_change") {
            payload.params = params;
        }

        formData.append("payload", JSON.stringify(payload));

        if (datasetFiles.length > 0 && feedbackType === "dataset_update") {
            datasetFiles.forEach((file) => formData.append("files", file));
        }

        try {
            const res = await fetch(`${process.env.REACT_APP_API_URL}/api/ml/feedback`, {
                method: "POST",
                body: formData
            });

            if (!res.ok) {
                const errorData = await res.json();
                throw new Error(errorData.detail || "Помилка сервера");
            }

            const data = await res.json();
            setFeedbackStatus(data);
            alert(`✅ Зміни типу "${feedbackType}" успішно надіслані!`);

            // Очищення полів після успішної відправки
            if (feedbackType === "parameter_change") {
                // Можливо, скинути до початкових або завантажити поточні з ML-сервісу
            }
            if (feedbackType === "dataset_update") {
                setDatasetFiles([]); // Очистити обрані файли
            }

        } catch (err) {
            console.error("Feedback error:", err);
            alert(`❌ Помилка при надсиланні змін: ${err.message}`);
        }
    };

    const handleApplyChangesAndRetrain = () => {
        alert("✅ Зміни параметрів збережено. Запуск перенавчання відбудеться у головному вікні.");
        navigate("/app", { state: { retrainStarted: true, modelName } });
    };


    // const handleApplyChangesAndRetrain = async () => {
    //     try {
    //         // Отримуємо поточні параметри моделі
    //         const currentModelParams = {
    //             test_size: params.test_size,
    //             max_iter: params.max_iter,
    //             C: params.C,
    //             solver: params.solver,
    //             model_name: modelName,
    //         };

    //         const res = await fetch(`${process.env.REACT_APP_API_URL}/api/ml/analyze`, {
    //             method: "POST",
    //             headers: { "Content-Type": "application/json" },
    //             body: JSON.stringify(currentModelParams),
    //         });

    //         if (!res.ok) {
    //             const errorData = await res.json();
    //             throw new Error(errorData.detail || "Не вдалося запустити перенавчання моделі");
    //         }

    //         const data = await res.json();
    //         alert("✅ Перенавчання моделі розпочато з урахуванням нових даних та параметрів!");
    //         console.log("Retrain started:", data);
    //         navigate("/app", { state: { retrainStarted: true, modelName } }); // Повернутися на головну сторінку, щоб бачити статус навчання

    //     } catch (err) {
    //         console.error("Retrain error:", err);
    //         alert(`❌ Помилка при перенавчанні моделі: ${err.message}`);
    //     }
    // };

    // const handleUpload = async () => {
    //     if (!files.length) {
    //         alert("Будь ласка, виберіть файли для завантаження");
    //         return;
    //     }
    //     const formData = new FormData();
    //     files.forEach((file) => formData.append("files", file));

    //     try {
    //         const res = await fetch(`${process.env.REACT_APP_API_URL}/api/ml/preprocess`, {
    //             method: "POST",
    //             body: formData,
    //         });

    //         if (!res.ok) throw new Error("Помилка сервера");

    //         const data = await res.json();
    //         console.log("✅ Файли надіслані:", data);
    //         alert("Файли успішно надіслані!");
    //     } catch (err) {
    //         console.error("Upload error:", err);
    //         alert("❌ Помилка відправки файлів");
    //     }
    // };

    // const handleSaveChanges = async () => {
    //     const formData = new FormData();
    //     formData.append("type", "parameter_change");
    //     formData.append("payload", JSON.stringify({ params }));

    //     datasetFiles.forEach((file) => formData.append("files", file));

    //     try {
    //         const res = await fetch(`${process.env.REACT_APP_API_URL}/api/ml/feedback`, {
    //             method: "POST",
    //             body: formData
    //         });

    //         const data = await res.json();
    //         setFeedbackStatus(data);
    //         alert("✅ Зміни успішно надіслані!");
    //     } catch (err) {
    //         console.error("Feedback error:", err);
    //         alert("❌ Помилка при надсиланні змін");
    //     }
    // };

    return (
        <div className="container mt-4">
            <h1 className="mb-4">Покращення моделі: {modelName}</h1>

            <h3>Параметри моделі</h3>
            <div className="row mb-3">
                {Object.entries(params).map(([key, value]) => (
                    <div key={key} className="col-md-3 mb-2">
                        <label className="form-label">{key}</label>
                        <input
                            type={typeof value === "number" ? "number" : "text"}
                            step={key === "test_size" || key === "C" ? "0.01" : "1"} // Для чисел з плаваючою крапкою
                            className="form-control"
                            value={value}
                            onChange={(e) => handleParamChange(key, e.target.value)}
                        />
                    </div>
                ))}
            </div>
            <button
                className="btn btn-primary mb-4"
                onClick={() => handleSubmitFeedback("parameter_change")}
            >
                Зберегти параметри
            </button>

            <h3>Додати нові дані</h3>
            <input type="file" multiple onChange={handleDatasetFileChange} className="form-control mb-2" />
            <button
                className="btn btn-primary mb-4"
                onClick={() => handleSubmitFeedback("dataset_update")}
                disabled={datasetFiles.length === 0}
            >
                Додати нові дані до датасету
            </button>


            <h3 className="mt-4">Застосувати зміни</h3>
            <button className="btn btn-success" onClick={handleApplyChangesAndRetrain}>
                Застосувати зміни та перенавчити модель
            </button>

            {feedbackStatus && (
                <pre className="mt-3 bg-light p-2 border rounded">
                    {JSON.stringify(feedbackStatus, null, 2)}
                </pre>
            )}
        </div>
    );
}
