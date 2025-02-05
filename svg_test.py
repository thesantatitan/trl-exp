import cairosvg

# Define a simple SVG image as a string
svg_content = '''
<svg>
  <rect x="50" y="50" width="200" height="200" stroke="#ff0000" stroke-width="10"/>
  <circle id="circle1" cx="50" cy="50" r="80" stroke="#ff0000" fill="none"/>
  <rect x="50" y="130" width="200" height="60" stroke="#ffd700" stroke-width="10"/>
  <circle id="circle2" cx="50" cy="50" r="80" stroke="#ffd700" stroke-width="10"/>
  <rect x="50" y="130" width="200" height="60" stroke="#ffd700" stroke-width="10"/>
  <circle id="circle3" cx="50" cy="50" r="80" stroke="#b28182" stroke-width="10"/>
  <rect x="50" y="130" width="200" height="60" stroke="#b28182" stroke-width="10"/>
  <circle id="circle4" cx="50" cy="50" r="80" stroke="#40e0c0" stroke-width="10"/>
  <rect x="50" y="130" width="200" height="60" stroke="#40e0c0" stroke-width="10"/>
  <circle id="circle5" cx="50" cy="50" r="80" stroke="#ff4444" stroke-width="10"/>
</svg>'''

# Output filename
output_file = "output.png"

# Convert SVG to PNG
cairosvg.svg2png(bytestring=svg_content.encode('utf-8'), write_to=output_file, parent_width=1024, parent_height=1024)

print(f"PNG file has been saved as {output_file}")
