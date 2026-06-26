"""
Pré-processamento e segmentação de imagens de cédulas.

Pipeline:
  1. Bilateral filter  — remove ruído preservando bordas
  2. Segmentação       — recorta/endireita a cédula (cor + bordas)
  3. CLAHE no canal L  — normaliza iluminação na região recortada
  4. Redimensionamento — padroniza tamanho para extração de features
"""

import cv2
import numpy as np

MIN_AREA_RATIO = 0.15
MAX_AREA_RATIO = 0.95
MIN_ASPECT = 1.5
MAX_ASPECT = 4.0
IDEAL_ASPECT = 2.4
MIN_SCORE = 0.25


def denoise(img, d=9, sigma_color=75, sigma_space=75):
    """Remove ruído JPEG/compressão sem borrar texturas finas."""
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def normalize_illumination(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Equalização adaptativa de contraste no canal L (espaço LAB)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge([l_channel, a_channel, b_channel])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _order_points(pts):
    """Ordena 4 pontos: topo-esq, topo-dir, baixo-dir, baixo-esq."""
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _is_valid_dimensions(bw, bh, img_w, img_h, area_ratio):
    """Valida se o recorte tem proporções compatíveis com uma cédula."""
    aspect = max(bw, bh) / max(min(bw, bh), 1)
    if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
        return False
    if area_ratio < MIN_AREA_RATIO or area_ratio > MAX_AREA_RATIO:
        return False
    if bw < img_w * 0.2 or bh < img_h * 0.2:
        return False
    return True


def _score_contour(contour, img_w, img_h):
    """Pontua um contorno: maior score = mais provável ser a cédula."""
    img_area = img_w * img_h
    area = cv2.contourArea(contour)
    if area <= 0:
        return -1

    x, y, bw, bh = cv2.boundingRect(contour)
    bbox_area_ratio = (bw * bh) / img_area
    if not _is_valid_dimensions(bw, bh, img_w, img_h, bbox_area_ratio):
        return -1

    aspect = max(bw, bh) / max(min(bw, bh), 1)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    quad_bonus = 0.25 if len(approx) == 4 else 0

    aspect_score = 1.0 - min(abs(aspect - IDEAL_ASPECT) / IDEAL_ASPECT, 1.0)
    area_score = 1.0 - min(abs(bbox_area_ratio - 0.45) / 0.45, 1.0)

    return (
        aspect_score * 0.35
        + area_score * 0.25
        + solidity * 0.15
        + quad_bonus
        + bbox_area_ratio * 0.1
    )


def _crop_from_contour(img, contour, pad_ratio=0.02):
    """Recorta região retangular em volta do contorno com padding."""
    h, w = img.shape[:2]
    x, y, bw, bh = cv2.boundingRect(contour)
    pad = max(2, int(pad_ratio * max(bw, bh)))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + bw + pad)
    y2 = min(h, y + bh + pad)
    return img[y1:y2, x1:x2]


def _warp_perspective(img, contour):
    """Endireita a cédula quando o contorno é um quadrilátero."""
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) != 4:
        return None

    pts = _order_points(approx.reshape(4, 2))
    width = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    height = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
    if width < 50 or height < 50:
        return None

    aspect = max(width, height) / max(min(width, height), 1)
    if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
        return None

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(img, matrix, (width, height))


def _collect_candidates(img, contours):
    """Gera candidatos de segmentação a partir de uma lista de contornos."""
    candidates = []
    h, w = img.shape[:2]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours[:8]:
        score = _score_contour(contour, w, h)
        if score < 0:
            continue

        warped = _warp_perspective(img, contour)
        if warped is not None:
            candidates.append({'score': score + 0.15, 'image': warped})
            continue

        cropped = _crop_from_contour(img, contour)
        if cropped.size > 0:
            candidates.append({'score': score, 'image': cropped})

    return candidates


def _segment_by_edges(img):
    """Segmentação via Canny + contornos."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    median = np.median(blurred)
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))
    edges = cv2.Canny(blurred, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return _collect_candidates(img, contours)


def _segment_by_color(img):
    """Segmentação via máscara de saturação (cédula colorida vs fundo neutro)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] > 40) & (hsv[:, :, 2] > 50)).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return _collect_candidates(img, contours)


def segment_banknote(img):
    """
    Segmenta a cédula combinando detecção por bordas e por cor.

    Escolhe o melhor candidato validado; se nenhum for confiável,
    retorna a imagem original (fallback seguro).
    """
    candidates = _segment_by_edges(img) + _segment_by_color(img)
    if not candidates:
        return img

    best = max(candidates, key=lambda c: c['score'])
    if best['score'] < MIN_SCORE:
        return img

    return best['image']


def build_hue_mask(img, sat_min=30, val_min=40):
    """
    Máscara de pixels com cor confiável para histograma HSV.

    Ignora regiões escuras ou pouco saturadas onde o Hue é instável.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return ((hsv[:, :, 1] > sat_min) & (hsv[:, :, 2] > val_min)).astype(np.uint8) * 255


def preprocess_image(img, target_size=(256, 256)):
    """
    Pipeline completo de pré-processamento e segmentação.

    Args:
        img: imagem BGR (numpy array) ou None
        target_size: tupla (largura, altura) para redimensionamento final

    Returns:
        Imagem processada ou None se a entrada for inválida
    """
    if img is None:
        return None

    img = denoise(img)
    img = segment_banknote(img)
    img = normalize_illumination(img)
    img = cv2.resize(img, target_size)

    return img
