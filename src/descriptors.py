"""
Extração de descritores para classificação de cédulas.

Descritores:
  - HSV Hue (com máscara de cor confiável)
  - HSV Saturação e Valor (bins reduzidos)
  - Hough Lines
  - LBP
"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern


def build_hue_mask(img, sat_min=30, val_min=40):
    """
    Máscara de pixels com cor confiável para histograma HSV.
    Ignora regiões escuras ou pouco saturadas onde o Hue é instável.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return ((hsv[:, :, 1] > sat_min) & (hsv[:, :, 2] > val_min)).astype(np.uint8) * 255


def extract_hue_histogram(img, bins=180):
    """Histograma de Hue ignorando pixels escuros ou pouco saturados."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = build_hue_mask(img)
    hist = cv2.calcHist([hsv[:, :, 0]], [0], mask, [bins], [0, 180])
    hist = hist.flatten().astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def extract_sv_histograms(img, bins=32):
    """Histogramas de Saturação e Valor — ajudam a separar azuis (R$50 vs R$100)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = build_hue_mask(img)
    s_hist = cv2.calcHist([hsv[:, :, 1]], [0], mask, [bins], [0, 256])
    v_hist = cv2.calcHist([hsv[:, :, 2]], [0], mask, [bins], [0, 256])
    s_hist = s_hist.flatten().astype(np.float32)
    v_hist = v_hist.flatten().astype(np.float32)
    for hist in (s_hist, v_hist):
        total = hist.sum()
        if total > 0:
            hist /= total
    return np.concatenate([s_hist, v_hist])


def extract_hough_features(img):
    """Estatísticas de linhas detectadas via Hough."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi / 180, threshold=100)

    if lines is None or len(lines) == 0:
        return np.zeros(5, dtype=np.float32)

    lines = lines[:, 0, :]
    theta_degrees = np.degrees(lines[:, 1]) % 180
    rho = lines[:, 0]

    return np.array([
        len(lines),
        np.mean(theta_degrees),
        np.std(theta_degrees),
        np.mean(np.abs(rho)),
        np.std(rho),
    ], dtype=np.float32)


def extract_lbp_histogram(img, radius=3, n_points=24, bins=59):
    """Histograma LBP uniforme."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins))
    hist = hist.astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def extract_all_features(img):
    """Concatena todos os descritores (308 features)."""
    return np.concatenate([
        extract_hue_histogram(img),      # 180
        extract_sv_histograms(img),      #  64
        extract_hough_features(img),     #   5
        extract_lbp_histogram(img),      #  59
    ])
