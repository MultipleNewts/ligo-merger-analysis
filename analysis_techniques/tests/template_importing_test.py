# %%
from analysis_techniques.templateImporting import Templates
import matplotlib.pyplot as plt
# %%
templates = Templates("templates/distance3000.json")
# Choose from distancexxxx.json where xxxx = 1128, 2000, 3000 or 4000.
print(templates.template_count)
# %%
template = templates.get_template(mass1=30, mass2=22)
plt.plot(template.times, template.hp)
plt.title(
    f"Mass1: {template.mass1}, "
    f"Mass2: {template.mass2}, "
    f"Distance: {template.distance}"
    )
plt.xlabel("Time")
plt.ylabel("Strain")
plt.show()
# %%
