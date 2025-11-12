// src/Router.jsx
import React, { useState } from "react";
import { BrowserRouter, Routes, Route, Navigate, useLocation } from "react-router-dom";
import AuthPage from "./AuthPage";
import App from "./App";
import ImproveModel from "./ImproveModel";

function ProtectedRoute({ user, children }) {
  const location = useLocation();
  if (!user) {
    return <Navigate to="/" state={{ from: location }} replace />;
  }
  return children;
}

export default function Router() {
  const [user, setUser] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem("user"));
    } catch {
      return null;
    }
  });

  return (
    <BrowserRouter>
      <Routes>
        {/* Сторінка авторизації */}
        <Route
          path="/"
          element={
            !user ? (
              <AuthPage setUser={setUser} />
            ) : (
              <Navigate to="/app" replace />
            )
          }
        />

        {/* Головна сторінка */}
        <Route
          path="/app"
          element={
            <ProtectedRoute user={user}>
              <App setUser={setUser} />
            </ProtectedRoute>
          }
        />

        {/* 👇 Додаємо новий маршрут для ImproveModel */}
        <Route
          path="/improve/:modelName"
          element={
            <ProtectedRoute user={user}>
              <ImproveModel />
            </ProtectedRoute>
          }
        />

        {/* Усі інші шляхи */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
