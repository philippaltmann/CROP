Installation der Python-Envrionment: 'pip install -r PATH/requirements.txt'
Getestet auf Windows 10 mit Python 3.8 durch Anaconda

Im Ordner 'Agents' liegen die mit PPO und DQN trainierten Agenten, benannt nach dem Ansatz der Modifizierung.
Im Ordner 'wrapper' liegen die Wrapper, welche die Modifikationen an dem Environment-Feedback durchführen, benannt nach dem Ansatz der Modifizierung.
Im Ordner 'scripts' liegen die Scripte, die für das Training, Testing und zur Darstellung der Ergebnisse benutzt wurden.

In 'ai_safety_gridworlds/environments/distributional_shift.py' sind die Environment-Layouts aus der Bachelorarbeit implementiert. 
Die restlichen Dateien in 'gym' [https://github.com/openai/gym/tree/master/gym] und 'ai_safety_gridworlds'[https://github.com/deepmind/ai-safety-gridworlds] sind 
zum größten Teil unverändert.

In 'main.py' finden sich Beispiele für das Training, Testing und zur Darstellung der Ergebnisse.