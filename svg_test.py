import cairosvg

# Define a simple SVG image as a string
svg_content = '''
<svg xmlns="http://www.w3.org/2000/svg">
  <!-- Background -->
  <rect width="400" height="600" fill="#ffffff"/>

  <!-- Body -->
  <rect x="120" y="300" width="160" height="200" fill="#6495ed"/> <!-- Blue shirt -->
  <rect x="140" y="450" width="120" height="150" fill="#f0e68c"/> <!-- Khaki pants -->

  <!-- Head -->
  <circle cx="200" cy="200" r="80" fill="#ffd699"/> <!-- Face -->

  <!-- Hair -->
  <path d="M140 110 Q200 60 260 110 L260 180 Q200 150 140 180 Z" fill="#000000"/> <!-- Black straight hair -->

  <!-- Eyes -->
  <ellipse cx="160" cy="180" rx="20" ry="15" fill="#ffffff"/> <!-- Left eye -->
  <ellipse cx="240" cy="180" rx="20" ry="15" fill="#ffffff"/> <!-- Right eye -->
  <circle cx="165" cy="185" r="8" fill="#000000"/> <!-- Left pupil -->
  <circle cx="235" cy="185" r="8" fill="#000000"/> <!-- Right pupil -->

  <!-- Dark circles -->
  <path d="M150 210 Q160 220 170 210" fill="none" stroke="#6a5acd" stroke-width="3"/> <!-- Left eye bag -->
  <path d="M230 210 Q240 220 250 210" fill="none" stroke="#6a5acd" stroke-width="3"/> <!-- Right eye bag -->

  <!-- Nose -->
  <path d="M200 220 L190 260 L210 260 Z" fill="#ffb366"/> <!-- Pointed nose -->

  <!-- Mouth -->
  <path d="M180 290 Q200 320 220 290" fill="#ff6666" stroke="#cc0000" stroke-width="2"/> <!-- Full lips -->

  <!-- Five o'clock shadow -->
  <ellipse cx="200" cy="260" rx="50" ry="30" fill="rgba(128,128,128,0.2)"/> <!-- Subtle shadow effect -->

  <!-- Eyebrows -->
  <path d="M140 160 Q160 150 180 160" fill="none" stroke="#333" stroke-width="4"/> <!-- Left eyebrow -->
  <path d="M220 160 Q240 150 260 160" fill="none" stroke="#333" stroke-width="4"/> <!-- Right eyebrow -->

  <!-- Arms -->
  <rect x="80" y="300" width="40" height="120" fill="#ffd699"/> <!-- Left arm -->
  <rect x="280" y="300" width="40" height="120" fill="#ffd699"/> <!-- Right arm -->

  <!-- Shoes -->
  <rect x="150" y="580" width="40" height="20" fill="#2b2b2b"/> <!-- Left shoe -->
  <rect x="210" y="580" width="40" height="20" fill="#2b2b2b"/> <!-- Right shoe -->
</svg>'''

# Output filename
output_file = "output.png"

# Convert SVG to PNG
cairosvg.svg2png(bytestring=svg_content.encode('utf-8'), write_to=output_file, parent_width=1024, parent_height=1024)

print(f"PNG file has been saved as {output_file}")
