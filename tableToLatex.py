def dict_to_latex_table(data):
    latex_table = "\\begin{tabular}{|c|c|c|c|}\n"
    latex_table += "\\hline\n"
    
    # Encabezado de la tabla
    headers = ["Modelo", "Directional Similarity", "Clip-I", "Clip-T"]
    latex_table += " & ".join(headers) + " \\\\\n"
    latex_table += "\\hline\n"
    
    # Filas de datos
    for model, values in data.items():
        row = [model.replace("_", "\\_")]
        row.extend([f"{value:.6f}" for value in values.values()])
        latex_table += " & ".join(row) + " \\\\\n"
    
    # Cierre de la tabla
    latex_table += "\\hline\n"
    latex_table += "\\end{tabular}"
    
    return latex_table

data = {
    "BLIP2_25Steps": {"Directional_similarity": 0.1256139513066749, "Clip-I": 0.212752065861348, "Clip-T": 0.24332491116425425},
    "BLIP2_50Steps": {"Directional_similarity": 0.12538532869525484, "Clip-I": 0.212752065861348, "Clip-T": 0.24312892187501967},
    "BLIP2_100Steps": {"Directional_similarity": 0.12742300222138153, "Clip-I": 0.212752065861348, "Clip-T": 0.24348349761717097},
    "BLIP2_200Steps": {"Directional_similarity": 0.1283732430106893, "Clip-I": 0.212752065861348, "Clip-T": 0.24382465432599648},
    "WO_50Steps": {"Directional_similarity": 0.061810590288380984, "Clip-I": 0.212752065861348, "Clip-T": 0.2374823786241492}   
}

latex_table = dict_to_latex_table(data)
print(latex_table)

'''
\begin{tabular}{|c|c|c|c|}
\hline
Modelo & Directional Similarity & Clip-I & Clip-T \\
\hline
BLIP2\_25Steps & 0.125614 & 0.212752 & 0.243325 \\
BLIP2\_50Steps & 0.125385 & 0.212752 & 0.243129 \\
BLIP2\_100Steps & 0.127423 & 0.212752 & 0.243483 \\
BLIP2\_200Steps & 0.128373 & 0.212752 & 0.243825 \\
WO\_50Steps & 0.061811 & 0.212752 & 0.237482 \\
\hline
\end{tabular}
'''