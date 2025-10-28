docker ps     
CONTAINER ID   IMAGE             COMMAND                  CREATED       STATUS       PORTS                                         NAMES
8d3bf070d701   mmsttsapi:local   "sh -c 'python -m uv…"   4 hours ago   Up 4 hours   0.0.0.0:8080->8080/tcp, [::]:8080->8080/tcp   mmsttsapi


docker stop 8d3bf070d701


python3 -m venv .venv
source .venv/bin/activate

brew install ffmpeg libsndfile

python3 -m pip install --upgrade pip
pip install -r requirements.txt -v

./run_local.sh

# dans un autre onglet
akourlaiev@ ~/workspace/mms-tts-api % curl http://localhost:8080/healthz
{"status":"ok","model":"facebook/mms-1b-all","lang_loaded":""}%


curl -X POST "http://localhost:8080/asr" \
  -F "lang=fra" \
  -F "file=@/Users/akourlaiev/workspace/mms-tts-api/fra.wav"

# Exemple en français (code MMS: fra)
curl -X POST "http://localhost:8080/asr" \
  -F "lang=fra" \
  -F "file=@/Users/akourlaiev/workspace/mms-tts-api/fra_long.wav"


akourlaiev@ ~/workspace/mms-tts-api % curl -X POST "http://localhost:8080/asr" \
  -F "lang=fra" \
  -F "file=@/Users/akourlaiev/workspace/mms-tts-api/fra_long.wav"

{"lang":"fra","text":"simor parle aiop est une varit du comorien qui lui-mame est une lande ban tout tr s proche du saili et lambe partage une origine comune et une grande partie du vocabulaire et de la gramaire et pendant simaor a evolue de maniere autonome avec des influences spcifiques notament de l'arabe du malgache et du franais et saili standard come celui parlet en ansani u ona utilise un lectique et des structures parfois difrentes dans une conversation un locuteur du simor reconaitra de nombrex mots et poura saisir l'ide genrale mais la comprhension complete sera dificile sans exposition pralable au sai","num_samples":668160,"sampling_rate":16000,"model":"facebook/mms-1b-all"}%  
=> > 1 min

akourlaiev@ ~/workspace/mms-tts-api % curl -X POST "http://localhost:8080/asr" \
  -F "lang=fra" \
  -F "file=@/Users/akourlaiev/workspace/mms-asr-api/data_test/Enregistrements/Artem/texte_1_AKO.wav"


docker run --rm -p 8080:8080 -e MODELS_DIR=/models -v "$(pwd)/models:/models" --name mmsasrapi mmsasrapi:local



top — tes logs sont très parlants 👇
Cold load modèle: ~11.9 s (download + init mms-1b-all, 3.86 GB).
I/O + préproc: ~0.18 s (decode, mono, resample, processor).
Forward (CPU): 28.0 s pour 48.65 s d’audio ⇒ RTF ≈ 0.58× (plus rapide que temps réel, mais RAM pic ~6.1 GB).
Decode CTC: 12 ms.
Ça confirme que le coût principal est le forward, avec un pic mémoire non trivial. Voilà comment réduire la latence et la RAM sans changer l’API.


#### test en local 
cd /Users/akourlaiev/workspace/mms-asr-api
source .venv/bin/activate

export HF_HOME="$(pwd)/models"

(.venv) akourlaiev@ ~/workspace/mms-asr-api % ./run_local.sh

Ce que disent tes métriques
Cold-load modèle : ~1.80 s (tu profites bien du cache HF_HOME local 👌).
Forward (CPU) : 20.08 s pour 48.65 s d’audio → RTF ≈ 0.41× (2.4× plus rapide que le temps réel).
Pic RAM : ~6.1 GB.
Le reste (I/O, resample, préproc, CTC) est négligeable.


##### test en local via Docker
docker run --rm -p 8080:8080 -e MODELS_DIR=/models -v "$(pwd)/models:/models" --name mmsasrapi mmsasrapi:local


#### test sur cloud run 
URL="https://mms-asr-api-615733745472.europe-west1.run.app"

# sans le auth
curl -v "$URL/healthz/"

# français
curl -v -X POST "$URL/asr" \
  -F "lang=fra" \
  -F "file=@./texte_1_AKO.wav"


URL="https://mms-asr-api-615733745472.europe-west1.run.app"
# français
curl -v -X POST "$URL/asr" \
  -F "lang=fra" \
  -F "file=@/Users/akourlaiev/workspace/mms-asr-api/data_test/fra.wav"


curl -v -X POST "https://mms-asr-api-615733745472.europe-west1.run.app/asr" \
  -F "lang=fra" \
  -F "file=@//Users/akourlaiev/workspace/mms-asr-api/data_test/Enregistrements/data_tests_dicte_v2_mono_8kHz_signed-16-bit-PCM-ALL-clean-v1/texte_1_AKO.wav"


gcloud run deploy mms-asr-api \
--project pole-emploi-trad-dev \
--image eu.gcr.io/pole-emploi-trad-dev/mms-asr-api \
--client-name "Cloud Code for VS Code" \
--client-version 2.31.1 \
--platform managed \
--region europe-west1 \
--port 8080 \
--cpu 4 \
--memory 16Gi \
--concurrency 2 \
--timeout 300 \
--service-account 615733745472-compute@developer.gserviceaccount.com \
--clear-env-vars

--gpu type=nvidia-tesla-t4,count=1 \
--set-env-vars=HF_HOME=/tmp/models,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1,TORCH_NUM_THREADS=1,MMS_MODEL_ID=facebook/mms-1b-all,LOG_LEVEL=INFO




# compression du modèle pour upload plus rapide sur GCS
akourlaiev@ ~/workspace/mms-asr-api/models/models % tar -cf - hub/models--facebook--mms-1b-all | zstd -3 -o mms-1b-all.tar.zs
/*stdin*\            : 59.07%   (  3.60 GiB =>   2.12 GiB, mms-1b-all.tar.zst) 



# copier sur GCS
akourlaiev@ ~/workspace/mms-asr-api % gsutil cp mms-1b-all.tar.zst gs://trademploi-models-dev/models/hub/


docker buildx build --platform linux/amd64 \
  --build-arg PT_DEVICE=cu121 \
  -t europe-west1-docker.pkg.dev/PROJECT/REPO/mms-asr-api:latest \
  --push .


docker buildx build --platform linux/amd64 \
  --build-arg PT_DEVICE=cpu \
  -t mmsasrapi:local .



{"lang":"fra","text":"vous souhaitez vous orienter vers les métiers d'aide a la personne il y a beaucoup d'offres dans ce domaine car c'est un métier an tention ses offres sont disponibles sur le site internet de France Travail je vous inscries une session de détection de potentiel et en fonction du bilan de la session nous verons ensemble les actions complémentaires mettre en place la session de DTP est prvue le jeudi x mai nev heure et quart au q rues du Général de Gaulle bâtiment bai premier étage dans le septime arrondissement de paris je vous informe de vos droits tes obligations","num_samples":603032,"sampling_rate":16000,"model":"facebook/mms-1b-all"}%   


desactivate



{"lang":"fra","text":"vous recherchez un poste de plombier j'ai bien not que vous ates autonome dans votre recherche d'emploi je vous afecte donc la modalit suivie vous recevrez le nom de votre conseiler parmel dans quelques jours vous alez publier votre sevele en ligne sur le site france travail et mettre jour vos comptences dans votre espace personnel je vous inscrit l'atelier de prsentation de la modalit suivie qui aura lieu le mars a dix heures en vision je vous done rendez-vous au job d'etin qui se droulera chez yokan le mai an heur tnt dans leur locaux a rosney sou bois je vous inscrit la prsentation active emploi je vous prsente l'emploi stor je vous romets une offre d'emploi je vous informe de vos droits et obligations","num_samples":778343,"sampling_rate":16000,"model":"facebook/mms-1b-all"}% 

{"lang":"fra","text":"vous avez perdu votre emploi la suite d'une rupture conventionele vous avez un bac pus q en gestion comptable vous etes cadre avec trois anes d'expriences profesioneles vous serez suivi par l'agence cadre de la paix i vous devez prendre contact avec l'agence d'ici le v avril pour fixer la date de votre entretien votre contact est m dubois il est disponible tous les jours sauf le mardi h a q heures","num_samples":469053,"sampling_rate":16000,"model":"facebook/mms-1b-all"}% 

{"lang":"fra","text":"vous avez besoin de clarifier votre projet profesionel je vous oriente vers un conseil rfrent cep le conseil en volution profesionele est un apui propose a tout actif qui souhaite faire le point sur sa situation et ventuelement voluer profesionelement c'est une offre de service gratuite vous alez participer a l'atelier conseil envisagez mon avenir profesionel pour identifier vos atouts comptences et intrats profesionels cet atelier poura tre aid construire un plan d'action pour realiser votre projet l'atelier a lieu l'undi prochain vous alez recevoir une convocation parmele vous devez l'imprimer et je vous informe de vos droits et devoir","num_samples":732832,"sampling_rate":16000,"model":"facebook/mms-1b-all"}%

{"lang":"fra","text":"vous vous intresez a la cration d'entreprises je vous invite vous inscrire gratuitement l'invenement go entrepreneur qui aura lieu le et q avril a paris la defense arena vous pouvez asister des confrences vous aurez acs des boites a outil tr s utiles vous ete intres par la rce pour recevoir l'aide la reprise et la cration d'entreprises vous devez remettre un france travail injustificatif atestant de la cration ou de la reprise d'une entreprise dans le cadre du dispositif acre come un extrais cabis","num_samples":575169,"sampling_rate":16000,"model":"facebook/mms-1b-all"}% 

{"lang":"fra","text":"vous venez de crer votre espace personnel sur le site france travail point t fer ce tableau de bord vous offre des nombreuses fonctionalits pour faciliter le suivi de vos recherches d'emploi et de vos dmarches administratives vous pouvez crer publie en ligne jusqu' cinq svs difrents vous avez la posibilit d'envoyer un mel votre conseiler ce service est disponible si vous confirmez votre adrese mel vous pouvez ainsi prendre rendez-vous en ligne avec votre conseiler je vous informe de vos droits et s'obligations","num_samples":526174,"sampling_rate":16000,"model":"facebook/mms-1b-all"}%  

{"lang":"fra","text":"c'est intermitent du spectacle je vous informe que dans votre espace personnel vous pouvez consulter l'historique de vos priode de travail maladies et formations des vig-qut dernires mois et vrifie que vos justificatifs ont bien t transmis vos employeurs France Travail vos employeurs ont jusqu'au qz du mois suivant de fin de contrat pour adreser les atestations employeurs ventueles em et qz jours aprs la reprsentation pour transmettre les dclarations uniques simplifies des us france travail ces informations sont disponibles dans la rubrique mon inscription puis mes deux dernires anes","num_samples":733529,"sampling_rate":16000,"model":"facebook/mms-1b-all"}%

{"lang":"fra","text":"vous etes intermitant du spectacle vous ralisez des contrats qui relvent du rgime gnral en parale de vos contrats au titre des anexes t vous indiquez avoir un contrat au rgime gnral au cours le jour de votre date aniversaire je vous informe que vous devez vous asurer d'avoir transmis l'intgralit de vos buletins de salaire a fance travail je vous rapele que vous devez procder l'actualisation mensuele du mois sur lequel se situe votre date aniversaire la priode d'actualisation ouvre le de chaque mois le pour le mois de fvrier et se terminent le du mois suivant","num_samples":682213,"sampling_rate":16000,"model":"facebook/mms-1b-all"}% 

{"lang":"fra","text":"vous etes intermitent du spectacle toute activit dclare a france travail doit tre justifie je vous informe des justificatifs fournir pour les activits ou l'vnement en dehors de vos contrats spectacle pour toute activit non salarie vous devez fournir votre dclaration de chifre d'afaires a l'ursaf et soit votre avis d'imposition soit si vous etes asujeti un impot sur les socits le proces verbal d'asemble gnrale","num_samples":463945,"sampling_rate":16000,"model":"facebook/mms-1b-all"}% 



curl -s "https://gpt-oss-20-b-615733745472.europe-west1.run.app/v1/chat/completions" \
  -H "Authorization: Bearer secret-xxxxxx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explique en une phrase ce qu’est gpt-oss-20b."}
    ],
    "temperature": 0.3,
    "max_tokens": 500
  }'



    "reasoning": { "effort": "low" },      // ⟵ réduit la réflexion

# avec la réflexion à faible effort
curl -s "https://gpt-oss-20-b-615733745472.europe-west1.run.app/v1/chat/completions" \
  -H "Authorization: Bearer secretemploi" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explique en une phrase ce qu’est gpt-oss-20b."}
    ],
    "reasoning": { "effort": "low" },
    "temperature": 0.3,
    "max_tokens": 256
  }' | jq .



python asr_batch.py \
  --dir "/Users/akourlaiev/workspace/mms-asr-api/data_test/Enregistrements/data_tests_dicte_v2_mono_8kHz_signed-16-bit-PCM-ALL-clean-v1" \
  --api-url "http://localhost:8080/asr" \
  --lang "fra" \
  --out "./asr_results.csv" \
  --pattern "**/*.wav" \
  --workers 1


python batch_asr_from_dir.py  \
  --dir "/Users/akourlaiev/workspace/mms-asr-api/data_test/Enregistrements/data_tests_dicte_v2_mono_8kHz_signed-16-bit-PCM-ALL-clean-v1" \
  --api-url "https://mms-asr-api-615733745472.europe-west1.run.app/asr" \
  --lang "fra" \
  --out "./asr_results.csv" \
  --pattern "*.wav" \
  --workers 1
