import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# Erstelle ein Verzeichnis, um die Bilder zu speichern
output_dir = '/mnt/c/dev/git/CAS_Applied_Data_Science/CAS_Final_Project/generated_images'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Funktion zum Generieren einer zufälligen 12-stelligen Zahl
def generate_random_number():
    return ''.join(random.choices('0123456789', k=12))


# Funktion zum Erstellen eines zufälligen Hintergrundes
def generate_random_background(width, height):
    img = Image.new('RGB', (width, height),
                    color=(random.randint(200, 255), random.randint(200, 255), random.randint(200, 255)))
    draw = ImageDraw.Draw(img)
    # Optional: Muster hinzufügen
    for _ in range(random.randint(5, 15)):
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        x2 = random.randint(0, width)
        y2 = random.randint(0, height)
        draw.line((x1, y1, x2, y2), fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
                  width=1)
    return img


# Funktion zum Laden einer Schriftart
def load_font(font_size):
    try:
        return ImageFont.truetype("/mnt/c/dev/git/CAS_Applied_Data_Science/CAS_Final_Project/fonts/arial_narrow_7.ttf", font_size)
    except IOError:
        return ImageFont.load_default()


for i in range(50):
    # Zufällige Bildgrösse
    img_width = random.randint(100, 400)
    img_height = random.randint(50, 100)

    # Neues Bild mit zufälligem Hintergrund erstellen
    img = generate_random_background(img_width, img_height)
    d = ImageDraw.Draw(img)

    # Zufällige 12-stellige Zahl generieren
    random_number = generate_random_number()

    # Startposition für den Text
    current_x = random.randint(10, 50)  # Zufällige Startposition mit etwas Rand
    current_y = random.randint(10, 100)  # Zufällige Startposition mit etwas Rand

    # Zufällige Grautonfarbe für die Ziffer
    gray_value = random.randint(0, 150)
    text_color = (gray_value, gray_value, gray_value)


    # Zufällige Schriftgrösse
    font_size = random.randint(8, 24)
    oldXoffset = 0
    for char in random_number:

        font = load_font(font_size)

        # Zufällige Verschiebung für jede Ziffer in y-Richtung
        #offset_x = random.randint(0, 20)
        offset_y = random.randint(-1, 1)

        # Zeichne die Ziffer mit zufälliger y-Verschiebung und Schriftgrösse
        d.text((current_x, current_y + offset_y), char, fill=text_color, font=font)

        # Aktualisiere die x-Position für die nächste Ziffer
        char_width = d.textbbox((0, 0), char, font=font)[2] - d.textbbox((0, 0), char, font=font)[0]
        current_x += char_width + random.randint(2, 10)  # Zufälliger Abstand zwischen den Ziffern

        # Überprüfe, ob die nächste Ziffer noch ins Bild passt
        if current_x + char_width > img_width:
            print("nuber not fully displayed in image")
            break
    # Zufällige Unschärfe anwenden
    blur_radius = random.uniform(0, 2)  # Radius zwischen 0 und 2
    img = img.filter(ImageFilter.GaussianBlur(blur_radius))

    # Bild speichern
    img.save(os.path.join(output_dir, f'image_{i:04d}.png'))

print("Bilder wurden erfolgreich generiert und gespeichert.")