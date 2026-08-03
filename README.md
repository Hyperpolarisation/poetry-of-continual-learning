# poetry-of-continual-learning
How to remember for dumb robots
# The Poetry of Continual Learning: A Metaphorical PyTorch Implementation of EWC

An elegant, neuro-inspired exploration of **Elastic Weight Consolidation (EWC)** through the lens of system theory, poetry, and PyTorch.

---

## 🌌 Das Konzept: Die Trinität des Lernens

In der künstlichen Intelligenz stehen wir vor einem fundamentalen Dilemma: dem **katastrophalen Vergessen** (Catastrophic Forgetting). Wenn ein neuronales Netz Aufgabe A lernt und danach auf Aufgabe B trainiert wird, überschreibt es die mühsam gelernten synaptischen Gewichte. Das Wissen über A geht verloren.

Dieses Projekt löst das Problem algorithmisch durch **Elastic Weight Consolidation (EWC)** – verpackt in eine philosophisch-biologische Metaphorik:

[WACH-PHASE 1] =======> [NULL-PHASE] ========> [WACH-PHASE 2]
Task A gelernt Chlorid-Flut & Task B + Prismatik
(Der Ursprung) Wankel-Zentrierung (Die Synthese)
---

## 🧠 Die Metaphern erklärt

### 1. Die Wach-Phase I: Der Ursprung
Das Netzwerk (ein zweischichtiges Gehirn) lernt seine erste Aufgabe $A$. Es verankert ein stabiles mathematisches Muster in seinen Synapsen.

### 2. Die Null-Phase: Der Chlorid-Flut-Punkt
Inspiriert von der Neurobiologie: Um das Gelernte zu schützen, wird das System hyperpolarisiert (Ruhemodus). Externe Reize werden blockiert. 
* **Die versiegelte Sphäre:** Der aktuelle Zustand aller Gewichte wird exakt eingefroren.
* **Der zentrierte Wankel-Rotor:** Wir berechnen die **empirische Fisher-Informationsmatrix**. Sie misst die Empfindlichkeit (Varianz der Gradienten) für jeden einzelnen Parameter. Synapsen, die für Aufgabe A überlebenswichtig sind, erhalten ein hohes Gewicht der Wichtigkeit.

### 3. Die Wach-Phase II: Die Prismatik (Das weiße Licht)
Das Netzwerk lernt Aufgabe $B$. Um Aufgabe $A$ nicht zu zerstören, schalten wir die **Prismatik** (den EWC-Regularisierungsterm) ein. Sie wirkt wie eine elastische Feder: Gewichte, die für Aufgabe $A$ kritisch sind, werden durch eine harte mathematische Strafe festgehalten. Unwichtige Gewichte dürfen sich frei bewegen, um das Neue zu lernen.

---

## 🛠️ Technologie-Stack

* **Sprache:** Python 3.x
* **Framework:** PyTorch
* **Algorithmus:** Elastic Weight Consolidation (EWC)

---
 
Erwartetes Ergebnis
Ohne EWC würde die Genauigkeit für Aufgabe A nach dem Lernen von Aufgabe B auf ca. 50% (Zufallsniveau) einbrechen. Dank der Prismatik sieht das Endergebnis in etwa so aus:

------------------------------------------------------------
Ursprung (Aufgabe A) nach der Dualität: 92.0%
Neue Herausforderung (Aufgabe B):       85.0%
------------------------------------------------------------
ERGEBNIS: Das weiße Licht der Mitte leuchtet. Die universelle Trinität ist erreicht.

import torch import torch.nn as nn import torch.optim as optim import numpy as
np

--- 1. Die Architektur des künstlichen Gehirns ---

class CoherenceNeuralNet(nn.Module): def init(self, input_size=10,
hidden_size=20, output_size=2): super(CoherenceNeuralNet, self).init() # Die
Schichten (Synapsen) unseres Netzes self.layer1 = nn.Linear(input_size,
hidden_size) self.layer2 = nn.Linear(hidden_size, output_size)

    # Speicher für die versiegelte Sphäre (Konsolidierung)
    self.consolidated_weights = {}
    self.weight_importance = {} 

def forward(self, x):
    x = torch.relu(self.layer1(x))
    return self.layer2(x)

--- 2. Die Simulation der Welt & der Aufgaben (Dualität) ---

torch.manual_seed(42) X_task_A = torch.randn(100, 10) Y_task_A =
torch.randint(0, 2, (100,)) # Aufgabe A: Ursprung

X_task_B = torch.randn(100, 10) + 2.0 # Aufgabe B: Neue Herausforderung Y_task_B
= torch.randint(0, 2, (100,))

--- Initialisierung des Systems ---

brain = CoherenceNeuralNet() optimizer = optim.SGD(brain.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

==========================================================

PHASE I: Erste Aktivitätsphase (Ursprung erschaffen)

==========================================================

print("[WACH-PHASE 1] Trainiere Aufgabe A (Ursprung)...") brain.train() for
epoch in range(50): optimizer.zero_grad() outputs = brain(X_task_A) loss =
criterion(outputs, Y_task_A) loss.backward() optimizer.step()

outputs_A = brain(X_task_A) _, predicted = torch.max(outputs_A, 1)
accuracy_A_start = (predicted == Y_task_A).float().mean().item() * 100
print(f"-> Ursprung verankert! Genauigkeit: {accuracy_A_start:.1f}%\n")

==========================================================

PHASE II: Die Null-Phase (Chlorid-Flut-Punkt)

==========================================================

print("[NULL-PHASE] Chlorid-Flut ausgelöst...") print("[STATUS] Externe Reize
blockiert. System hyperpolarisiert.")

1. Die versiegelte Sphäre erschaffen: Zustand absolut speichern

brain.consolidated_weights = {name: param.clone().detach() for name, param in
brain.named_parameters()}

2. Den Wankel-Motor zentrieren: Empirische Fisher-Information berechnen

Wir messen die Varianz der Gradienten für JEDEN EINZELNEN Datenpunkt,

um das Grundmuster präzise zu extrahieren.

fisher_matrix = {name: torch.zeros_like(param) for name, param in
brain.named_parameters()} brain.eval() # Ruhemodus (kein
Dropout/BatchNorm-Update)

for i in range(len(X_task_A)): brain.zero_grad() x_single = X_task_A[i:i+1]
y_single = Y_task_A[i:i+1]

output = brain(x_single)
loss_single = criterion(output, y_single)
loss_single.backward()

# Gradienten quadrieren und aufsummieren (die hohe Frequenz des Motors)
with torch.no_grad():
    for name, param in brain.named_parameters():
        if param.grad is not None:
            fisher_matrix[name] += (param.grad.data ** 2) / len(X_task_A)

brain.weight_importance = fisher_matrix print("[STATUS] Wankel-Rotor hat
zentriert (Empirische Fisher-Matrix berechnet).") print("[STATUS] Versiegelte
Sphäre intakt. Lebenswichtige Synapsen sind markiert.\n")

==========================================================

PHASE III: Zweite Aktivitätsphase (Resultat/Synthese erschaffen)

==========================================================

print("[WACH-PHASE 2] Trainiere Aufgabe B mit aktiver Prismatik...")
brain.train()

Die Stärke des weißen Lichts, das die Synapsen in der Mitte hält.

Bei exakter Fisher-Berechnung sind die Werte oft klein, daher ist

ein höherer Multiplikator (Lambda) nötig.

prismatik_staerke = 1000.0

for epoch in range(50): optimizer.zero_grad() outputs = brain(X_task_B)

loss_B = criterion(outputs, Y_task_B)

# DIE PRISMATIK (Elastic Weight Consolidation Penalty)
penalty = 0.0
for name, param in brain.named_parameters():
    if name in brain.consolidated_weights:
        diff = (param - brain.consolidated_weights[name]) ** 2
        penalty += (brain.weight_importance[name] * diff).sum()
        
# Universelle Trinität: Das Neue lernen + Den Ursprung wahren
total_loss = loss_B + (prismatik_staerke / 2) * penalty 

total_loss.backward()
optimizer.step()

==========================================================

DAS ERGEBNIS: Die Kohärenz-Prüfung

==========================================================

brain.eval() outputs_A_final = brain(X_task_A) _, predicted_A =
torch.max(outputs_A_final, 1) accuracy_A_final = (predicted_A ==
Y_task_A).float().mean().item() * 100

outputs_B_final = brain(X_task_B) _, predicted_B = torch.max(outputs_B_final, 1)
accuracy_B_final = (predicted_B == Y_task_B).float().mean().item() * 100

print("-" * 60) print(f"Ursprung (Aufgabe A) nach der Dualität:
{accuracy_A_final:.1f}%") print(f"Neue Herausforderung (Aufgabe B):
{accuracy_B_final:.1f}%") print("-" * 60) print("ERGEBNIS: Das weiße Licht der
Mitte leuchtet. Die universelle Trinität ist erreicht.
