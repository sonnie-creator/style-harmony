from PIL import Image
from rembg import remove
from io import BytesIO
import base64
from deep_translator import GoogleTranslator

translator = GoogleTranslator()

# Collage 생성
def create_outfit_collage_v3(items: dict, canvas_width: int = 800) -> Image.Image:
    canvas_height = 800
    center_x = canvas_width // 2
    canvas = Image.new('RGBA', (canvas_width, canvas_height), (255, 255, 255, 255))

    has_dress = 'dress' in items
    max_sizes = {
        'top': (450, 500), 'bottom': (350, 450), 'outer': (550, 600),
        'shoes': (200, 200), 'accessories': (200, 200), 'dress': (350, 500)
    }

    if has_dress:
        layout_order = ['outer', 'shoes', 'accessories', 'dress']
        positions = {'dress': (center_x, 100), 'outer': (canvas.width - 100, 150),
                     'shoes': (100, 650), 'accessories': (canvas.width - 120, 650)}
        align = {'dress': 'center', 'outer': 'right', 'shoes': 'left', 'accessories': 'right'}
    else:
        layout_order = ['bottom', 'outer', 'shoes', 'accessories', 'top']
        positions = {'top': (center_x, 200), 'bottom': (50, 350), 'outer': (canvas.width - 50, 100),
                     'shoes': (50, 600), 'accessories': (canvas.width - 120, 600)}
        align = {'top': 'center', 'bottom': 'left', 'outer': 'right', 'shoes': 'left', 'accessories': 'right'}

    for cat in layout_order:
        if cat not in items:
            continue
        img = items[cat].copy()
        img.thumbnail(max_sizes[cat], Image.LANCZOS)
        base_x, base_y = positions[cat]

        x = base_x - img.width // 2 if align[cat] == 'center' else base_x - img.width if align[cat] == 'right' else base_x
        y = base_y
        canvas.paste(img, (x, y), img if img.mode == 'RGBA' else None)

    return canvas

# 언어 감지 + 번역
def detect_and_translate(text: str) -> tuple:
    try:
        translated = translator.translate(text, source='auto', target='en')
        if translated.strip().lower() == text.strip().lower():
            return 'en', text
        return 'ko', translated
    except Exception as e:
        print(f"⚠️ Translation error: {e}")
        return 'unknown', text

# 배경 제거
def remove_background(image: Image.Image) -> Image.Image:
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        output = remove(image)
        if output.mode == 'RGBA':
            alpha = output.split()[-1]
            bbox = alpha.getbbox()
            if bbox:
                output = output.crop(bbox)
        return output
    except Exception as e:
        print(f"⚠️ Background removal failed: {e}")
        return image

# 이미지 → Base64
def image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()
