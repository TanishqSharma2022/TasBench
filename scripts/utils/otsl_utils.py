#!/usr/bin/env python3
"""
OTSL to HTML conversion utility.
Parses both character-based OTSL (C, F, L, U, X, N) and token-based OTSL (<ched>, <fcel>, <lcel>, <ucel>, <xcel>, <nl>).
"""
import re

def normalize_otsl(otsl_text):
    """Normalize any OTSL string or token sequence to standard characters:
       C (Cell), L (Left span / colspan continuation), U (Up span / rowspan continuation),
       X (Cross span / block continuation), N (New line)
    """
    if isinstance(otsl_text, list):
        otsl_text = " ".join(otsl_text)
    
    # Replace verbose tokens
    text = otsl_text.replace("<ched>", " C ").replace("<fcel>", " C ").replace("<ecel>", " C ")
    text = text.replace("<lcel>", " L ").replace("<ucel>", " U ").replace("<xcel>", " X ")
    text = text.replace("<nl>", " N ")
    text = text.replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", "")
    text = text.replace("C-tag", " C ").replace("L-tag", " L ").replace("NL-tag", " N ")
    text = text.replace("<thead>", "").replace("</thead>", "").replace("<tbody>", "").replace("</tbody>", "")
    
    # Keep only letters
    tokens = text.split()
    chars = []
    for tok in tokens:
        for ch in tok:
            ch_up = ch.upper()
            if ch_up in ['C', 'F', 'E']:
                chars.append('C')
            elif ch_up == 'L':
                chars.append('L')
            elif ch_up == 'U':
                chars.append('U')
            elif ch_up == 'X':
                chars.append('X')
            elif ch_up == 'N':
                chars.append('N')
    
    # Ensure it ends with N
    if chars and chars[-1] != 'N':
        chars.append('N')
    return "".join(chars)

def otsl_to_html(otsl_text):
    otsl_norm = normalize_otsl(otsl_text)
    if not otsl_norm:
        return "<table></table>"
    
    rows_raw = otsl_norm.strip().split('N')
    rows = [list(r) for r in rows_raw if r]
    if not rows:
        return "<table></table>"
    
    R = len(rows)
    C = max(len(r) for r in rows) if rows else 0
    if C == 0:
        return "<table></table>"
    
    # Pad all rows to C
    grid = []
    for r in rows:
        if len(r) < C:
            r = r + ['C'] * (C - len(r))
        grid.append(r[:C])

    html = ["<table>"]
    for i in range(R):
        html.append("<tr>")
        for j in range(C):
            cell = grid[i][j]
            if cell == 'C':
                # Check colspan
                colspan = 1
                while j + colspan < C and grid[i][j + colspan] == 'L':
                    colspan += 1
                # Check rowspan
                rowspan = 1
                while i + rowspan < R and grid[i + rowspan][j] == 'U':
                    rowspan += 1
                
                cs_str = f' colspan="{colspan}"' if colspan > 1 else ""
                rs_str = f' rowspan="{rowspan}"' if rowspan > 1 else ""
                html.append(f"<td{rs_str}{cs_str}></td>")
            # L, U, X cells are consumed by their parent cell's span
        html.append("</tr>")
    html.append("</table>")
    return "".join(html)

if __name__ == "__main__":
    test_otsl = "FFFNFFFN"
    print("Test HTML:", otsl_to_html(test_otsl))
