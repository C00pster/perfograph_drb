import perfograph as pg
import os

error_files = []

for filename in os.listdir("samples"):
    if (filename[-4:] == ".cpp"):
        try:
            G = pg.from_file("samples/" + filename, disable_progress_bar=True)
            pg.to_json(G, "perfograph/" + filename[:-4] + ".json")
        except:
            error_files.append(filename)
    elif (filename[-2:] == ".c"):
        try:
            G = pg.from_file("samples/" + filename, disable_progress_bar=True)
            pg.to_json(G, "perfograph/" + filename[:-2] + ".json")
        except:
            error_files.append(filename)

print("Error files: ", error_files)