from PIL import Image, ImageDraw

def make_circle_logo(input_path, output_path):
    # Open the image
    img = Image.open(input_path).convert("RGBA")
    
    # Create a circular mask
    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)
    width, height = img.size
    
    # Draw a white circle on the mask (this will be the opaque part)
    # We leave a small margin to ensure clean edges if desired, or go full edge
    draw.ellipse((0, 0, width, height), fill=255)
    
    # Apply the mask
    result = Image.new("RGBA", img.size, (0, 0, 0, 0))
    result.paste(img, (0, 0), mask=mask)
    
    # Save
    result.save(output_path, "PNG")
    print(f"Created circular logo at {output_path}")

if __name__ == "__main__":
    make_circle_logo("dashboard/logo.png", "dashboard/logo_circle.png")
