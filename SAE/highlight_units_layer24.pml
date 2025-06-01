reinitialize
fetch 4WXQ, async=0
hide everything, 4WXQ
show cartoon, 4WXQ
set cartoon_smooth_loops, 1
set cartoon_transparency, 0.1
select keep_ions, (4WXQ and (resn NA+CA))
show spheres, keep_ions
hide labels, keep_ions
select just_protein, 4WXQ and not keep_ions
color slate, just_proteinset sphere_scale, 0.4, keep_ions

# Layer 24, Unit 2739, Positive
select layer24_unit_2739_residues, resi 290+301+394+405+408+438+440+467+488
color red, layer24_unit_2739_residues
zoom complete
png layer24_unit_2739.png, dpi=300, ray=1
save layer24_unit_2739.pse
color lightblue, layer24_unit_2739_residues
deselect

# Layer 24, Unit 1809, Positive
select layer24_unit_1809_residues, resi 292+294+300+331+333+341+343+351+353+398+410+412+433+435+445+457+459+483+485+489+491+493+495
color red, layer24_unit_1809_residues
zoom complete
png layer24_unit_1809.png, dpi=300, ray=1
save layer24_unit_1809.pse
color lightblue, layer24_unit_1809_residues
deselect

# Layer 24, Unit 3024, Positive
select layer24_unit_3024_residues, resi 250+309
color red, layer24_unit_3024_residues
zoom complete
png layer24_unit_3024.png, dpi=300, ray=1
save layer24_unit_3024.pse
color lightblue, layer24_unit_3024_residues
deselect

# Layer 24, Unit 3078, Positive
select layer24_unit_3078_residues, resi 350
color red, layer24_unit_3078_residues
zoom complete
png layer24_unit_3078.png, dpi=300, ray=1
save layer24_unit_3078.pse
color lightblue, layer24_unit_3078_residues
deselect

# Layer 24, Unit 660, Positive
select layer24_unit_660_residues, resi 249+256+300+305+307+356+359+361+363+410+412+433
color red, layer24_unit_660_residues
zoom complete
png layer24_unit_660.png, dpi=300, ray=1
save layer24_unit_660.pse
color lightblue, layer24_unit_660_residues
deselect

# Layer 24, Unit 3379, Negative
select layer24_unit_3379_residues, resi 436+448+450+473+486+496
color red, layer24_unit_3379_residues
zoom complete
png layer24_unit_3379.png, dpi=300, ray=1
save layer24_unit_3379.pse
color lightblue, layer24_unit_3379_residues
deselect

# Layer 24, Unit 3147, Negative
select layer24_unit_3147_residues, resi 248+289+392+440
color red, layer24_unit_3147_residues
zoom complete
png layer24_unit_3147.png, dpi=300, ray=1
save layer24_unit_3147.pse
color lightblue, layer24_unit_3147_residues
deselect

# Layer 24, Unit 2631, Negative
select layer24_unit_2631_residues, resi 249+251+269+288+291+295+298+315+328+344+354+383+390+402+426+439+449+495
color red, layer24_unit_2631_residues
zoom complete
png layer24_unit_2631.png, dpi=300, ray=1
save layer24_unit_2631.pse
color lightblue, layer24_unit_2631_residues
deselect

# Layer 24, Unit 2913, Negative
select layer24_unit_2913_residues, resi 298+319+322+328+371+373+376+383+403+417+421+431+467+471+473+475+478
color red, layer24_unit_2913_residues
zoom complete
png layer24_unit_2913.png, dpi=300, ray=1
save layer24_unit_2913.pse
color lightblue, layer24_unit_2913_residues
deselect

# Layer 24, Unit 3186, Negative
select layer24_unit_3186_residues, resi 398
color red, layer24_unit_3186_residues
zoom complete
png layer24_unit_3186.png, dpi=300, ray=1
save layer24_unit_3186.pse
color lightblue, layer24_unit_3186_residues
deselect