import streamlit as st
import cv2
import numpy as np
import io
from PIL import Image


st.set_page_config(page_title="Aplikasi Pengolahan Citra Digital", layout="wide")

st.title("📸 Aplikasi Pengolahan Citra Digital")
st.write("Unggah gambar dan pilih jenis pengolahan yang ingin dilakukan.")

uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Baca gambar
    image = Image.open(uploaded_file).convert("RGB")  # pastikan RGB
    img_array = np.array(image)
    adjusted = img_array.copy()

    st.sidebar.title("⚙️ Pengaturan Fitur")

    # ======================  BASIC ADJUSTMENT ======================
    st.sidebar.header("Basic Adjustment")

    if st.sidebar.checkbox("Brightness & Contrast"):
        brightness = st.sidebar.slider("Tingkat Brightness", -100, 100, 0)
        contrast = st.sidebar.slider("Tingkat Kontras", -100, 100, 0)
        adjusted = cv2.convertScaleAbs(adjusted, alpha=1 + (contrast / 100), beta=brightness)

    if st.sidebar.checkbox("Grayscale"):
        gray = cv2.cvtColor(adjusted, cv2.COLOR_RGB2GRAY)
        adjusted = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    if st.sidebar.checkbox("Histogram Equalization"):
        gray = cv2.cvtColor(adjusted, cv2.COLOR_RGB2GRAY)
        equalized = cv2.equalizeHist(gray)
        adjusted = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)

    # ====================== NOISE & EDGE ======================
    st.sidebar.header("Noise & Edge")

    if st.sidebar.checkbox("Hilangkan Noise (Median Filter)"):
        kernel = st.sidebar.slider("Ukuran Kernel", 1, 15, 3, step=2)
        adjusted = cv2.medianBlur(adjusted, kernel)

    if st.sidebar.checkbox("Edge Detection"):
        low_threshold = st.sidebar.slider("Low Threshold", 0, 255, 50)
        high_threshold = st.sidebar.slider("High Threshold", 0, 255, 150)
        gray_for_edge = cv2.cvtColor(adjusted, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray_for_edge, low_threshold, high_threshold)
        adjusted = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    if st.sidebar.checkbox("Non-Local Means Denoising"):
        h_value = st.sidebar.slider("H (Filter Strength)", 1, 30, 10)
        template_window = st.sidebar.slider("Ukuran Template Window", 3, 15, 7, step=2)
        search_window = st.sidebar.slider("Ukuran Search Window", 5, 35, 21, step=2)
        adjusted = cv2.fastNlMeansDenoisingColored(adjusted, None, h_value, h_value, template_window, search_window)

    # ====================== TRANSFORMASI GAMBAR ======================
    st.sidebar.header("Transformasi Gambar")

    rotate_quick = st.sidebar.selectbox("Rotasi Cepat", ["Tidak", "90°", "180°", "270°"])
    if rotate_quick == "90°":
        adjusted = cv2.rotate(adjusted, cv2.ROTATE_90_CLOCKWISE)
    elif rotate_quick == "180°":
        adjusted = cv2.rotate(adjusted, cv2.ROTATE_180)
    elif rotate_quick == "270°":
        adjusted = cv2.rotate(adjusted, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if st.sidebar.checkbox("Rotasi Manual"):
        angle = st.sidebar.slider("Sudut Rotasi (°)", -180, 180, 0)
        if angle != 0:
            h, w = adjusted.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            cos = np.abs(matrix[0, 0])
            sin = np.abs(matrix[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
            matrix[0, 2] += (new_w / 2) - center[0]
            matrix[1, 2] += (new_h / 2) - center[1]
            adjusted = cv2.warpAffine(adjusted, matrix, (new_w, new_h))

    flip_option = st.sidebar.selectbox("Flip", ["Tidak", "Horizontal", "Vertikal", "Keduanya"])
    if flip_option == "Horizontal":
        adjusted = cv2.flip(adjusted, 1)
    elif flip_option == "Vertikal":
        adjusted = cv2.flip(adjusted, 0)
    elif flip_option == "Keduanya":
        adjusted = cv2.flip(adjusted, -1)

    if st.sidebar.checkbox("Crop Gambar"):
        h, w = adjusted.shape[:2]
        top = st.sidebar.slider("Potong Atas", 0, h // 2, 0)
        bottom = st.sidebar.slider("Potong Bawah", 0, h // 2, 0)
        left = st.sidebar.slider("Potong Kiri", 0, w // 2, 0)
        right = st.sidebar.slider("Potong Kanan", 0, w // 2, 0)
        y1, y2 = top, h - bottom
        x1, x2 = left, w - right
        if y1 < y2 and x1 < x2:
            adjusted = adjusted[y1:y2, x1:x2]
        else:
            st.warning("Nilai crop terlalu besar, gambar tidak valid!")

    # ====================== COLOR MANIPULATION ======================
    st.sidebar.header("Color Manipulation")
    color_option = st.sidebar.selectbox(
        "Pilih metode:",
        ["Tidak", "Channel Split & Merge", "Ubah Saturasi dan Hue", "Color Space Conversion"]
    )

    if color_option == "Channel Split & Merge":
        r, g, b = cv2.split(img_array)
        merged = cv2.merge([r, g, b])
        st.sidebar.markdown("**Tampilkan channel RGB secara terpisah atau gabungkan kembali.**")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.image(r, caption="Channel R", use_column_width=True, clamp=True)
        with col2:
            st.image(g, caption="Channel G", use_column_width=True, clamp=True)
        with col3:
            st.image(b, caption="Channel B", use_column_width=True, clamp=True)
        with col4:
            st.image(merged, caption="Merged RGB", use_column_width=True, clamp=True)

    elif color_option == "Ubah Saturasi dan Hue":
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        hue_shift = st.sidebar.slider("Geser Hue (warna)", -180, 180, 0)
        saturation_scale = st.sidebar.slider("Skala Saturasi", 0.0, 3.0, 1.0)
        hsv[..., 0] = (hsv[..., 0].astype(int) + hue_shift) % 180
        hsv[..., 1] = np.clip(hsv[..., 1].astype(float) * saturation_scale, 0, 255).astype(np.uint8)
        adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    elif color_option == "Color Space Conversion":
        space = st.sidebar.selectbox("Pilih ruang warna:", ["HSV", "LAB", "YCrCb"])
        if space == "HSV":
            converted = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            c1, c2, c3 = cv2.split(converted)
            labels = ["Hue", "Saturation", "Value"]
            back_conv = cv2.COLOR_HSV2RGB
        elif space == "LAB":
            converted = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            c1, c2, c3 = cv2.split(converted)
            labels = ["L (Lightness)", "A (Green-Red)", "B (Blue-Yellow)"]
            back_conv = cv2.COLOR_LAB2RGB
        else:
            converted = cv2.cvtColor(img_array, cv2.COLOR_RGB2YCrCb)
            c1, c2, c3 = cv2.split(converted)
            labels = ["Y (Luma)", "Cr (Red Diff)", "Cb (Blue Diff)"]
            back_conv = cv2.COLOR_YCrCb2RGB

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(c1, caption=labels[0], use_column_width=True, clamp=True)
        with col2:
            st.image(c2, caption=labels[1], use_column_width=True, clamp=True)
        with col3:
            st.image(c3, caption=labels[2], use_column_width=True, clamp=True)
        st.image(cv2.cvtColor(converted, back_conv),
                 caption=f"Gambar dalam ruang warna {space}",
                 use_column_width=True)

    # ====================== TAMPILKAN HASIL ======================
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_array, caption="Gambar Asli", use_container_width=True)
    with col2:
        st.image(adjusted, caption="Hasil Edit", use_container_width=True)

    # pastikan ada variabel hasil untuk disimpan/display selanjutnya
    img_result = adjusted.copy()

    # ============================== SIMPAN / UNDUH GAMBAR ==============================

    st.markdown("---")
    st.subheader("💾 Simpan / Unduh Gambar")

    if 'img_result' in locals() and img_result is not None:
        col1, col2 = st.columns(2)
        with col1:
            file_format = st.radio("Pilih format file:", ["JPG", "PNG"], horizontal=True)
        with col2:
            file_name = st.text_input("Nama file (tanpa ekstensi):", "hasil_edit")
        pil_img = Image.fromarray(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))

        buffer = io.BytesIO()
        ext = "JPEG" if file_format == "JPG" else "PNG"
        pil_img.save(buffer, format=ext)
        buffer.seek(0)

        # Tombol download
        st.download_button(
            label=f"Unduh Gambar ({file_format})",
            data=buffer,
            file_name=f"{file_name}.{file_format.lower()}",
            mime=f"image/{file_format.lower()}"
        )
    else:
        st.info("⚠️ Tidak ada gambar yang bisa disimpan. Silakan lakukan pengeditan terlebih dahulu.")

else:
    st.info("Silakan unggah gambar terlebih dahulu untuk mulai mengedit.")
