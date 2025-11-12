import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

export default function AuthPage({ setUser }) {
    const [activeTab, setActiveTab] = useState("login");
    const [loginData, setLoginData] = useState({ email: "", password: "" });
    const [registerData, setRegisterData] = useState({
        name: "",
        email: "",
        password: "",
        org: "",
        role: "user",
    });
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const navigate = useNavigate();

    const apiBase = process.env.REACT_APP_API_URL || "http://localhost:5000";

    const handleLogin = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError("");
        try {
            const res = await fetch(`${apiBase}/api/ml/login`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(loginData),
            });
            const data = await res.json();
            if (res.ok && data.status === "authenticated") {
                localStorage.setItem("user", JSON.stringify(data.user));
                setUser(data.user);
                navigate("/app", { replace: true });
            } else {
                setError(data.detail || "Невірний email або пароль");
            }
        } catch (err) {
            console.error(err);
            setError("Помилка підключення до сервера");
        } finally {
            setLoading(false);
        }
    };

    const handleRegister = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError("");
        try {
            const res = await fetch(`${apiBase}/api/ml/register`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(registerData),
            });
            const data = await res.json();
            if (res.ok && data.status === "registered") {
                alert("✅ Реєстрація успішна! Тепер увійдіть у свій акаунт.");
                setActiveTab("login");
            } else {
                setError(data.detail || "Не вдалося зареєструватись");
            }
        } catch (err) {
            console.error(err);
            setError("Помилка з'єднання з сервером");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="d-flex justify-content-center align-items-center vh-100 bg-light">
            <div className="card shadow p-4" style={{ width: "400px" }}>
                <h3 className="text-center mb-3">📰 Fake News Detection</h3>

                <ul className="nav nav-tabs mb-3 justify-content-center">
                    <li className="nav-item">
                        <button
                            className={`nav-link ${activeTab === "login" ? "active" : ""}`}
                            onClick={() => setActiveTab("login")}
                        >
                            Вхід
                        </button>
                    </li>
                    <li className="nav-item">
                        <button
                            className={`nav-link ${activeTab === "register" ? "active" : ""}`}
                            onClick={() => setActiveTab("register")}
                        >
                            Реєстрація
                        </button>
                    </li>
                </ul>

                {error && (
                    <div className="alert alert-danger py-2 text-center">{error}</div>
                )}

                {activeTab === "login" ? (
                    <form onSubmit={handleLogin}>
                        <div className="mb-3">
                            <label className="form-label">Email</label>
                            <input
                                type="email"
                                className="form-control"
                                required
                                value={loginData.email}
                                onChange={(e) =>
                                    setLoginData({ ...loginData, email: e.target.value })
                                }
                            />
                        </div>
                        <div className="mb-3">
                            <label className="form-label">Пароль</label>
                            <input
                                type="password"
                                className="form-control"
                                required
                                value={loginData.password}
                                onChange={(e) =>
                                    setLoginData({ ...loginData, password: e.target.value })
                                }
                            />
                        </div>
                        <button
                            type="submit"
                            className="btn btn-primary w-100"
                            disabled={loading}
                        >
                            {loading ? "Вхід..." : "Увійти"}
                        </button>
                    </form>
                ) : (
                    <form onSubmit={handleRegister}>
                        <div className="mb-2">
                            <label className="form-label">Ім’я</label>
                            <input
                                type="text"
                                className="form-control"
                                required
                                value={registerData.name}
                                onChange={(e) =>
                                    setRegisterData({ ...registerData, name: e.target.value })
                                }
                            />
                        </div>
                        <div className="mb-2">
                            <label className="form-label">Email</label>
                            <input
                                type="email"
                                className="form-control"
                                required
                                value={registerData.email}
                                onChange={(e) =>
                                    setRegisterData({ ...registerData, email: e.target.value })
                                }
                            />
                        </div>
                        <div className="mb-2">
                            <label className="form-label">Пароль</label>
                            <input
                                type="password"
                                className="form-control"
                                required
                                value={registerData.password}
                                onChange={(e) =>
                                    setRegisterData({
                                        ...registerData,
                                        password: e.target.value,
                                    })
                                }
                            />
                        </div>
                        <div className="mb-2">
                            <label className="form-label">Організація (необов’язково)</label>
                            <input
                                type="text"
                                className="form-control"
                                value={registerData.org}
                                onChange={(e) =>
                                    setRegisterData({ ...registerData, org: e.target.value })
                                }
                            />
                        </div>
                        <div className="mb-3">
                            <label className="form-label">Роль</label>
                            <select
                                className="form-select"
                                value={registerData.role}
                                onChange={(e) =>
                                    setRegisterData({ ...registerData, role: e.target.value })
                                }
                            >
                                <option value="user">Користувач</option>
                                <option value="researcher">Дослідник</option>
                            </select>
                        </div>
                        <button
                            type="submit"
                            className="btn btn-success w-100"
                            disabled={loading}
                        >
                            {loading ? "Реєстрація..." : "Зареєструватися"}
                        </button>
                    </form>
                )}
            </div>
        </div>
    );
}
