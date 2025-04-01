def gerar_telemetria_video(output, video, *telemetria, **kw):
	"""
	Gera um vídeo composto por:
		- A parte esquerda: o vídeo original (possivelmente redimensionado)
		- A parte direita: gráficos animados que exibem as telemetria empilhados verticalmente
	
	Parâmetros:
		output: string com o nome do arquivo de saída.
		video: string com o nome do arquivo de vídeo original.
		telemetria: cada telemetria pode ser:
			- Um DataFrame ou Series (que será convertido, se necessário), ou
			- Uma tupla no formato (telemetria, kwt) onde kwt é um dicionário com opções de plot.
	
	Kwargs (chaves permitidas):
		temp_dir: (opcional) diretório onde os arquivos auxiliares serão criados; se não informado, usa o diretório atual.
		atraso: (opcional) tempo (em segundos) que indica quando as telemetrias iniciam (referenciado ao instante zero do vídeo).
		span: (opcional) largura (em segundos) do eixo x dos gráficos, com 0 no centro.
		dpi: (opcional) DPI usado para gerar os gráficos.
		xlabel: (opcional) rótulo do eixo x.
		deleteaux: (opcional) se True, remove os arquivos auxiliares após a execução.
	Se forem encontrados kwargs desconhecidos, a função lançará um erro.
	"""
	#%% ----------------------  Tabela descritiva dos comportamentos da função para diferentes configurações de timestamps e atraso  -----------------------
	#	caso	|		timestamp_video		|		timestamp_tel		|		atraso		|	Descrição
	#			|							|							|					|
	#	1		|		não informado		|		não informado		|			0		|	Assumir tudo zero
	#	2		|		não informado		|		não informado		|	   positivo		|	Telemetria começa depois do vídeo
	#	3		|		não informado		|		não informado		|	   negativo		|	Vídeo começa depois da telemetria
	#	4		|		não informado		|		  informado			|			0		|	Assume o timestamp do vídeo igual ao da telemetria
	#	5		|		não informado		|		  informado			|	   positivo		|	Calcula o timestamp do vídeo subtraindo o atraso do timestamp da telemetria
	#	6		|		não informado		|		  informado			|	   negativo		|	Calcula o timestamp do vídeo subtraindo o atraso do timestamp da telemetria
	#	7		|		  informado			|		não informado		|			0		|	Assume o timestamp da telemetria igual ao do vídeo 
	#	8		|		  informado			|		não informado		|	   positivo		|	Calcula o timestamp da telemetria somando o atraso ao timestamp do vídeo
	#	9		|		  informado			|		não informado		|	   negativo		|	Calcula o timestamp da telemetria somando o atraso ao timestamp do vídeo
	#	10		|		  informado			|		  informado			|			0		|	Calcula o atraso total (timestamp_tel - timestamp_video + atraso informado), telemetria começa neste atraso depois do vídeo
	#	11		|		  informado			|		  informado			|	   positivo		|	Calcula o atraso total (timestamp_tel - timestamp_video + atraso informado), telemetria começa neste atraso depois do vídeo
	#	12		|		  informado			|		  informado			|	   negativo		|	Calcula o atraso total (timestamp_tel - timestamp_video + atraso informado), telemetria começa neste atraso depois do vídeo

	#%% Imports, definição de funções auxiliares
	import os
	import ffmpeg
	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
	from matplotlib.colors import is_color_like as iscolor
	import warnings
	import re
	import inspect
	import time
	from tqdm import tqdm
	import psutil
	import threading

	def calcular_largura_yticklabels(*tels, **kwargs):
		"""
		Calcula a largura máxima necessária para acomodar os yticklabels de todas as telemetrias passadas,
		considerando as configurações de fonte especificadas em plotkw (como ytick_fontsize, fontweight, fontfamily, fontstyle).
		
		Cada telemetria pode ser:
			- Um DataFrame ou Series (que será convertido se necessário), ou
			- Uma tupla no formato (telemetria, kw) onde kw é um dicionário que pode conter a chave "plotkw"
			com configurações de fonte para os yticklabels.
		
		Retorna:
			A largura máxima em pixels.
		"""
		import matplotlib.pyplot as plt
		from matplotlib.font_manager import FontProperties
		import matplotlib.backends.backend_agg as agg
		import pandas as pd
		import re
		
		kwargs=list(kwargs.items())[0][1]
		dpi=kwargs['dpi']
		max_width = 0
		fp = FontProperties()  # Propriedades padrão
		for tel in tels:
			tel=tel[0]
			if isinstance(tel, tuple): 			data = tel[0]; kw = tel[1]
			else: 								data = tel[0]; kw = {}
			if isinstance(data, pd.Series): 	data = data.to_frame()
			# Extrai configurações de fonte se existirem
			font_kwargs = {}
			if isinstance(kw, dict) and "plotkw" in kw:
				plotkw = kw["plotkw"]
				for key in ["ytick_fontsize", "fontweight", "fontfamily", "fontstyle"]:
					if key in plotkw:
						font_kwargs[key] = plotkw[key]
			if font_kwargs:
				fp = FontProperties(**font_kwargs)
			fig, ax = plt.subplots(dpi=dpi)
			if not data.empty:
				y_min = data.min().min()
				y_max = data.max().max()
				ax.set_ylim(y_min, y_max)
			else:
				ax.set_ylim(0, 1)
			fig.canvas.draw()
			canvas = agg.FigureCanvasAgg(fig)
			canvas.draw()
			renderer = canvas.get_renderer()
			pad_pts = plt.rcParams['ytick.major.pad']  # Padrão do Matplotlib
			pad_px = pad_pts * fig.dpi / 72  # Convertendo de pontos para pixels
			minus_width, _, _ = renderer.get_text_width_height_descent("−", fp, ismath=False)
			# for label in ax.get_yticklabels():
			# 	text = label.get_text()
			# 	if not text:
			# 		continue
			# 	label.set_fontproperties(fp)
			# 	bbox = label.get_window_extent(renderer=renderer)
			# 	width = bbox.width
			# 	if width > max_width:
			# 		max_width = width
			for label in ax.get_yticklabels():
				text = label.get_text()
				if not text: continue
				label.set_fontproperties(fp)
				w, _, _ = renderer.get_text_width_height_descent(text, fp, ismath=False)
				if "−" in text:  # Matplotlib usa U+2212 "−", que é diferente de hífen "-"
					w += minus_width
			width_total = w + pad_px
			if width_total > max_width:
					max_width = width_total
			plt.close(fig)
		return max_width
	def calcular_altura_xticklabels(**kwargs):
		"""
		Calcula a altura extra necessária para os xticklabels e xlabel, considerando os kwargs passados.
		"""
		kwargs=list(kwargs.items())[0][1]
		fig, ax = plt.subplots(dpi=kwargs['dpi'])
		altura_extra = 0
		
		# Verifica se xlabel está presente nos kwargs e calcula altura
		if "xlabel" in kwargs:
			label = kwargs["xlabel"] if isinstance(kwargs["xlabel"], str) else "Exemplo"
			xlabel_text = ax.set_xlabel(label)
			fig.canvas.draw()
			altura_extra += xlabel_text.get_window_extent().height / fig.dpi
		
		# Cria xticklabels fictícios para calcular a altura ocupada
		ax.set_xticks([0, 1, 2])
		ax.set_xticklabels(["0", "1", "2"])
		fig.canvas.draw()
		
		# Calcula altura máxima dos xticklabels
		ticklabel_heights = [tick.get_window_extent().height / fig.dpi for tick in ax.get_xticklabels()]
		if ticklabel_heights:
			altura_extra += max(ticklabel_heights)
		
		plt.close(fig)
		return altura_extra
	def safe_remove(filepath, retries=5, delay=0.2):
		for i in range(retries):
			try:
				os.remove(filepath)
				return
			except Exception as e:
				time.sleep(delay)
		warnings.warn(f"Não foi possível remover o arquivo {filepath} após {retries} tentativas.")
	def make_frame(tempo, *tels, **kwargs):
		"""
		Gera frames dos gráfico de telemetria:
		"""
		# print(f"Gerando frame para o tempo t={tempo}")
		kwargs=list(kwargs.items())[0][1]
		tels=tels[0]
		fig, ax = plt.subplots(nrows=len(tels),
			figsize=(telemetry_area_width/kwargs["dpi"], altura/kwargs["dpi"]),
			dpi=kwargs["dpi"],
			constrained_layout=False,
			sharex=True
		)
		fig.subplots_adjust(hspace=kwargs["hspace"], **bordas_plot)  # Reduz o espaço vertical entre os gráficos
		timestamp_tel = kwargs['timestamp_tel']
		
		tt=0	# Cria o índice tt fora do for para ser acessível depois de finalizadas as iterações
		for tt, tel in enumerate(tels):
			# Extrai os argumentos em plotkw e legendkw
			kwt = tel[1];	tel=tel[0]
			legendkw = kwt["legendkw"]		# Argumentos em legendkw
			plotkw = kwt["plotkw"]			# Argumentos em plotkw
			labels = plotkw["labels"]
			coltolabel_parser = plotkw["coltolabel_parser"]			

			# Calcula a máscara de tempos para plotar no gráfico
			janela = (tel.index >= tempo - kwargs["span"]/2) & (tel.index <= tempo + kwargs["span"])
			if len(janela)==0: 				continue


			# Plota todas as colunas passadas na telemetria tel
			for cc, col in enumerate(tel.columns):
				if labels is not None:								lab = labels[cc] if cc < len(labels) else col	# Usa labels explícitos 
				elif coltolabel_parser is not None:
					if callable(coltolabel_parser):					lab = coltolabel_parser(col);					# Usa coltolabel_parser como uma função
					else: m = re.search(coltolabel_parser, col); 	lab = m.group(1) if m else col					# Usa coltolabel_parser como um regex
				else: 												lab = col										# Condição padrão, usa o nome da coluna como label

				# Calcula o valor instantâneo de col no instante 0, indicado pelo índice central de janela
				indices_janela = tel.index[janela]
				if not (indices_janela <= tempo).any(): val = np.nan 	# Se não houverem valores à esquerda de ou coincidindo com a linha de zero, faz val=np.nan que sinaliza para que não seja impresso valor instantâneo
				else:
					i_central = (len(indices_janela) - 1) >> 1  # Se n for ímpar, i_central aponta exatamente para o centro. Divisão inteira por 2 equivale a deslocamento de um bit para a direita
					if (len(indices_janela) & 1) == 0 and indices_janela[i_central] > tempo:	i_central -= 1	# Se n for par, i é truncado; então, se indices_janela[i] for maior que tempo, ajusta para i-1
					val = tel.loc[indices_janela[i_central], col]

				# Plota col
				ax[tt].plot(indices_janela - tempo, tel.loc[janela,col],
					label = printyvals_label(lab,val,algar[tt]), 
					**dict(filter(filter_plotkw,plotkw.items()))
				)

			# Ajusta outros elementos gráficos
			ax[tt].set_ylim(	kwt["ylim"] if "ylim" in kwt.keys() 	else	[tel.min().min(),tel.max().max()]	)	 	
			ax[tt].axvline(0, color="red", linewidth=1)
			ax[tt].legend(**{i:legendkw[i] for i in legendkw if i!="unitlabel"})	# a função filter_legendkw não funcionou, filtrou todos os kwargs
			if facecolor is not None: ax[tt].set_facecolor(facecolor)
			ax[tt].grid(which='major', color='#e0e0e0', linewidth=1.5)
			ax[tt].grid(which='minor', color='#f0f0f0', linewidth=1)
			ax[tt].tick_params(axis='y', labelsize=kwargs['fontsize_ticklabel'])
		ax[tt].tick_params(axis='x', labelsize=kwargs['fontsize_ticklabel'])
		ax[tt].set_xticks(np.linspace(-kwargs["span"]/2, kwargs["span"]/2, num=5))
		ax[tt].set_xlim(-kwargs["span"]/2, kwargs["span"]/2)
		ax[tt].set_xlabel(xlabel,loc='right',fontsize=kwargs['fontsize_xlabel'])
		ax[tt].annotate(f"timestamp (tel) = {timestamp_tel}  {"" if tempo<0 else "+"}{tempo:.{casasdec_tempo}f}",fontsize=kwargs['fontsize_timestamp'],
			xy=(0,0), xycoords='figure pixels',
			xytext=(0,0), textcoords='offset pixels', ha='left', va='bottom',
			bbox=dict(boxstyle='square,pad=1', fc='none', ec='none')
        )
		canvas = FigureCanvas(fig)
		canvas.draw()
		img = np.array(canvas.renderer.buffer_rgba())
		plt.close(fig)
		return img



	#%% ==========================================  Validação e tratamento de argumentos, processamentos iniciais  =========================================
	err = lambda msg: (_ for _ in ()).throw(ValueError(msg));	inst=isinstance
	printyvals_fmt=lambda num,tam,dec: "" if np.isnan(num) else f"{num:>{tam}.{dec}f}"	# Função de format da string para uso nas funções printyvals_...
	printyvals_label=lambda lab,val: lab						# Compõe a string com nome da série, e o valor instantâneo quando especificado que este seja mostrado no label.
	comp_texto_vals=[0]*len(telemetria)							# Maior comprimento da string resultante da formatação de cada telemetria, considerando todas as colunas
	algar=[0]*len(telemetria)									# Número de algarismos necessários para imprimir a maior string resultante da conversão, considerando fundos de escala

	# Cria máscaras para argumentos de funções padrões. A função filter é lazy, isto é, processa chamadas quando e na medida que for requisitada, ao invés de processar todo o eval no momento da chamada. 
	filter_plotkw = lambda key_value: key_value[0] in frozenset(inspect.signature(plt.plot).parameters.keys())		
	filter_legendkw = lambda key_value: key_value[0] in frozenset(inspect.signature(plt.legend).parameters.keys())

	# -------------------------------------------------------  argumentos de definição do vídeo  -----------------------------------------------------------
	videokw={}
	if isinstance(video,tuple):
		if len(video)!=2:																	raise ValueError(f"Video: tupla deve ter exatamente 2 elementos.")
		video,videokw = video
		if not isinstance(video,str):														raise ValueError(f"Video: Quando definido em tuple, o primeiro elemento deve ser uma string com diretório/nome.")
		if not isinstance(videokw,dict):													raise ValueError(f"Video: Quando definido em tuple, o segundo elemento deve ser um dicionário.")

		videokw_copy=videokw.copy()
		args=[]
		arg = "timestamp_video";	videokw[arg] = timestamp_video = videokw.get(arg, None);		err(f"{videokw[arg]} deve ser um número") 		if not inst(videokw[arg], (int,float)) and videokw[arg] is not None			else args.append(arg)
		timestamp_video = float(timestamp_video)
		

		for arg in args: videokw_copy.pop(arg) if arg in videokw_copy else None
		if bool(videokw_copy): 																raise ValueError(f"Kwarg inesperado nos argumentos do vídeo: {list(videokw_copy)}")

	else: 
		if not isinstance(video, str):
			raise ValueError(f"Video: Quando definido em tuple, o segundo elemento deve ser um dicionário.")
		else:
			timestamp_video = None
			videokw=dict(timestamp_video=timestamp_video)

	# -------------------------------------------------------------  kwargs gerais da função  -------------------------------------------------------------- 
	# Verifica e extrai valores dos kwargs. O procedimento é feito desta forma para que a extração, e a verificação de kwargs desconhecidos,
	# seja feita definindo os kwargs permitidos numa única lista estática. Para adicionar novos parâmetros execute os passos abaixo: 
	# 1: implemente a utilização do novo parâmetro no corpo da função; 2 adicione uma linha nova copiando uma já criada; 
	# 3: especifique o nome da variável como string em arg; 4: altere o nome da variável conforme implementado em 1; 5: escreva msg e condição para validação.
	# Se uma variável implicar numa inicialização, dela mesma ou de outra variável, complexa demais para a função .get(), adicione como bloco de instruções abaixo
	kw_copy=kw.copy()
	args=[];	
	# 3:  "nome"----------|	4: 		  	  nome = kw.get(arg,valorpadrao)----------|	5:  msg---------------------------------------------------------cond--------------------------------------------------------continuação----------
	arg = "timestamp_tel";		kw[arg] = timestamp_tel = kw.get(arg, None);		err(f"{kw[arg]} deve ser um número") 							if not inst(kw[arg], (int,float)) and kw[arg] is not None	else args.append(arg)
	arg = "videowidth";			kw[arg] = videowidth = kw.get(arg, 0.6);			err(f"{kw[arg]} deve ser um float entre (0,1), não inclusive") 	if not inst(kw[arg], float) or kw[arg]<=0 or kw[arg]>=1		else args.append(arg)
	arg = "atraso";				kw[arg] = atraso = kw.get(arg, 0);					err(f"{kw[arg]} deve ser um número") 							if not inst(kw[arg], (int,float)) 							else args.append(arg)
	arg = "span";				kw[arg] = kw.get(arg, 10);							err(f"{kw[arg]} deve ser um número maior que zero") 			if not inst(kw[arg], (int,float)) or kw[arg]<=0 			else args.append(arg)

	arg = "dpi";				kw[arg] = kw.get(arg, 100);							err(f"{kw[arg]} deve ser um número maior que zero") 			if not inst(kw[arg], (int,float)) or kw[arg]<=0 			else args.append(arg)
	arg = "xlabel";				kw[arg] = xlabel = kw.get(arg, None);				err(f"{kw[arg]} deve ser uma string") 							if not inst(kw[arg], str) and kw[arg] is not None			else args.append(arg)
	arg = "hspace";				kw[arg] = kw.get(arg, 0.06);						err(f"{kw[arg]} deve ser um número") 							if not inst(kw[arg], (int,float)) 							else args.append(arg)
	arg = "facecolor";			kw[arg] = facecolor = kw.get(arg, None);			err(f"{kw[arg]} deve ser uma cor em formato do matplotlib") 	if not iscolor(kw[arg]) and kw[arg] is not None 			else args.append(arg)
	arg = "margem_ext";			kw[arg] = margem_ext = kw.get(arg, 10);				err(f"{kw[arg]} deve ser um número") 							if not inst(kw[arg], (int,float)) 							else args.append(arg)
	arg = "fontsize_ticklabel";	kw[arg] = kw.get(arg, 14);							err(f"{kw[arg]} deve ser um número >=0") 						if not inst(kw[arg], (int,float)) or kw[arg]<0				else args.append(arg)
	arg = "fontsize_xlabel";	kw[arg] = kw.get(arg, 16);							err(f"{kw[arg]} deve ser um número >=0") 						if not inst(kw[arg], (int,float)) or kw[arg]<0				else args.append(arg)
	arg = "fontsize_timestamp";	kw[arg] = kw.get(arg, 12);							err(f"{kw[arg]} deve ser um número >=0") 						if not inst(kw[arg], (int,float)) or kw[arg]<0				else args.append(arg)
	arg = "fontsize_legend";	kw[arg] = kw.get(arg, 8);							err(f"{kw[arg]} deve ser um número >=0") 						if not inst(kw[arg], (int,float)) or kw[arg]<0				else args.append(arg)
	# arg = "margem_int";			kw[arg] = margem_int = kw.get(arg, 30);				err(f"{kw[arg]} deve ser um número") 							if not inst(kw[arg], (int,float)) 							else args.append(arg)
	arg = "casasdec_tempo";		kw[arg] = casasdec_tempo = kw.get(arg, 2);			err(f"{kw[arg]} deve ser um inteiro >=0") 						if not inst(kw[arg], int) or kw[arg]<0 						else args.append(arg)
	arg = "casasdec_valor";		kw[arg] = casasdec_valor = kw.get(arg, 1);			err(f"{kw[arg]} deve ser um inteiro >=0") 						if not inst(kw[arg], int) or kw[arg]<0 						else args.append(arg)
	arg = "printyvals";			kw[arg] = printyvals = kw.get(arg, False);			err(f"{kw[arg]} deve ser False ou uma string: 'legend'") 		if not inst(kw[arg], str) and kw[arg]!=False				else args.append(arg)
	if printyvals=='legend':
		printyvals_label = lambda lab,val,tam:		lab + " " + printyvals_fmt(val,tam,casasdec_valor).rjust(tam)

	arg = "temp_dir"; 			kw[arg] = temp_dir = kw.get(arg, "" if os.name == "nt" else "./"); 		err(f"{kw[arg]} deve ser uma string contendo um diretório") 	if not inst(kw[arg], str) 									else args.append(arg)
	arg = "deleteaux";			kw[arg] = deleteaux = kw.get(arg, True);								err(f"{kw[arg]} deve ser um bool") 								if not inst(kw[arg], bool) 									else args.append(arg)
	# bool(dict) retorna verdadeiro se o dicionário não está vazio. Os argumentos válidos e esperados 
	# foram processados e removidos de kw_copy acima. Se sobrar algum argumento, ele será inválido
	for arg in args: kw_copy.pop(arg) if arg in kw_copy else None
	if bool(kw_copy): 																		raise ValueError(f"Kwarg inesperado: {list(kw_copy)}")
	del arg, args, kw_copy

	kw=kw | videokw
	algar = list(map(lambda x: x + casasdec_valor, algar))


	# ------------------------------------------------------------  telemetrias e seus kwargs  -------------------------------------------------------------
	# Verificação de estruturas inesperadas e de erros no passamento da variável telemetria. Depende de 
	if not telemetria: 																		raise ValueError("Nenhuma telemetria foi passada.")
	aux = [];	
	for tt, tel in enumerate(telemetria):
		if not (isinstance(tel, tuple) or isinstance(tel, (pd.DataFrame, pd.Series))):		raise ValueError("Telemetrias devem ser especificadas como um DataFrame, Series ou (DataFrame/Series, dict).")
		if isinstance(tel, tuple):
			if len(tel) != 2: 																raise ValueError(f"Telemetria {tt}: tupla deve ter exatamente 2 elementos.")
			tel_data, kwt = tel
			
			if isinstance(tel_data, pd.Series): tel_data = tel_data.to_frame()		# Converte Series para DataFrame
			if not isinstance(kwt, dict):													raise ValueError(f"Telemetria {tt}: quando passado como tupla, o segundo elemento deve ser um dicionário.")

			# Trata os argumentos de legendkw
			arg = "legendkw";				kwt[arg] = legendkw = kwt.get(arg, {"ncol": len(tel_data.columns), "loc": "upper left"})
			if not (isinstance(legendkw, dict)):											raise ValueError(f"Telemetria {tt}: legendkw deve ser um dicionário.")
			arg = "unitlabel";				legendkw[arg] = legendkw.get(arg, None)
			arg = "fontsize";				legendkw[arg] = legendkw.get(arg, kw["fontsize_legend"]);			err(f"{legendkw[arg]} deve ser um número >=0") 						if not inst(legendkw[arg], (int,float)) or legendkw[arg]<0				else None

			# Trata os argumentos de plotkw
			arg = "plotkw";					kwt[arg] = plotkw = kwt.get(arg, {})
			if not (isinstance(plotkw, dict)):												raise ValueError(f"Telemetria {tt}: plotkw deve ser um dicionário.")
			arg = "labels";					plotkw[arg] = labels = plotkw.get(arg, None)
			arg = "coltolabel_parser";		plotkw[arg] = coltolabel_parser = plotkw.get(arg, None)
			if (labels is not None) and (coltolabel_parser is not None):
				warnings.warn(f"labels e coltolabel_parser fornecidos para a telemetria de índice {tt}. O parser será ignorado.")
				plotkw["coltolabel_parser"]=None
		else:
			tel_data = tel
			kwt = {}
			if isinstance(tel_data, pd.Series): tel_data = tel_data.to_frame()		# Converte Series para DataFrame
		aux.append((tel_data, kwt))

		# Calcula o maior número de algarismos necessários para imprimir o fundo de escala de tel
		algar[tt]=max(  [len(str(round(tel_data.min().min()))),len(str(round(tel_data.max().max())))]  )

		# Calcula o índice de tel onde ocorre o valor instantâneo que leva ao maior comprimento de string quando renderizado em texto
		id=[tel_data.idxmin().min(),tel_data.idxmax().max()]
		tam=[len(printyvals_fmt(tel_data.loc[id[0]].iloc[0],10,10)),len(printyvals_fmt(tel_data.loc[id[1]].iloc[0],10,10))]		# Usa um valor fixado de tam e dec. Qual número leva ao maior string não depende dessas quantidades
		comp_texto_vals[tt]=id[tam.index(max(tam))]
	telemetria=aux

	# ------------------------------------------------------------  Extrai informações do vídeo  -----------------------------------------------------------
	probe = ffmpeg.probe(video)
	video_stream = next(s for s in probe["streams"] if s["codec_type"]=="video")
	largura = int(video_stream["width"])
	altura = int(video_stream["height"])
	duracao = float(video_stream["duration"])
	fps = eval(video_stream["avg_frame_rate"])
	
	# -------------------------------------- Divisão de largura, cálculos de dimensões, timestamps e atraso --------------------------------------
	video_area_width = int(videowidth * largura)
	telemetry_area_width = largura - video_area_width
	margem_esq=calcular_largura_yticklabels(telemetria, kwargs=kw)
	margem_inf=calcular_altura_xticklabels(kwargs=kw)
	bordas_plot=dict(	
						left=(margem_esq+margem_ext*2.1)/telemetry_area_width,
						right=(telemetry_area_width-margem_ext)/telemetry_area_width,
						top=(altura-margem_ext)/altura,
						# bottom=(margem_inf)/altura 	
					)

	atraso_total=0
	if timestamp_video is not None and timestamp_tel is not None: 		atraso_total=timestamp_tel-timestamp_video
	atraso_total+=atraso
	atraso_total = round(atraso_total, casasdec_tempo)
	# Depois de arredondado, crimpa atraso_total para zero quando dentro de uma tolerância para evitar imprimir timestamp +-0.00...
	if abs(atraso_total) < 1*10**(-casasdec_tempo+1): 					atraso_total = 0




	#%% ======================================  Processa os frames em pipe para gerar o vídeo auxiliar da telemetria  ======================================
	# temp_file_tel = os.path.join(temp_dir, f"telemetria_aux.mp4")
	# if os.path.exists(temp_file_tel): safe_remove(temp_file_tel)
	process_tel = (
		ffmpeg
		.input('pipe:0',
			format='rawvideo',
			pix_fmt='rgba',
			s=f'{telemetry_area_width}x{altura}',
			framerate=fps,
			thread_queue_size=1024  # Aumenta o tamanho da fila de threads
			)
		.output('pipe:1',
				format='rawvideo',
				pix_fmt='rgba',
				r=fps,
				max_muxing_queue_size=1024  # Aumenta o tamanho da fila de multiplexação
			)
		.overwrite_output()
		.run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
	)
	# # Verificar se o processo foi iniciado corretamente
	# print("Processo ffmpeg iniciado...")

	# --------------------------------------------------------  Inicia o monitoramento do processo  --------------------------------------------------------
	# def monitor_ffmpeg_stderr():
	# 	while True:
	# 		output = process_tel.stderr.readline()
	# 		if output:
	# 			print(f"[FFmpeg]: {output.decode().strip()}")
	# 		if process_tel.poll() is not None:
	# 			break
	# threading.Thread(target=monitor_ffmpeg_stderr, daemon=True).start()

	# ----------------------------------------------------------------  Geração dos frames  ----------------------------------------------------------------
	n_frames = int(duracao * fps)
	def frame_generator():
		for frame_idx in range(n_frames):
			t = frame_idx / fps + atraso_total
			# print(f"Gerando frame {frame_idx} com t={t}")
			img = make_frame(t, telemetria, kwargs=kw)
			frame_data = img.tobytes()
			
			# Verificando o conteúdo dos frames gerados
			# print(f"Frame {frame_idx}: {frame_data[:10]}...")  # Exibe apenas os primeiros 100 bytes para inspeção
			
			yield frame_data
	# Produz uma sequência de frames no formato .raw para fins de depuração
	# for frame_idx in range(n_frames):
	# 	# Gera o frame
	# 	t = frame_idx / fps + atraso_total
	# 	frame_data = make_frame(t, telemetria, kwargs=kw)
	# 	with open(f"frame{frame_idx:03d}.raw", "wb") as f:
	# 		f.write(frame_data)



	for frame_idx, frame_data in enumerate(tqdm(frame_generator(), total=n_frames, desc="Processando frames"), start=1):
		try:
			process_tel.stdin.write(frame_data)
		except Exception as e:
			warnings.warn(f"Erro ao escrever frame {frame_idx}: {e}")
			break	
		except BrokenPipeError:
			print("Erro: ffmpeg fechou o pipe inesperadamente.")
			break
	process_tel.stdin.close()
	process_tel.wait()
	time.sleep(0.5)



	#%% ========================================================== Compõe o vídeo e a telemetria ==========================================================
	final = ffmpeg.filter([
		ffmpeg.input(video).filter("scale", video_area_width, altura).output("pipe:", format="rawvideo", pix_fmt="rgba"),
		ffmpeg.input("pipe:", format="rawvideo", pix_fmt="rgba", s=f"{telemetry_area_width}x{altura}", framerate=fps)
	], "hstack", inputs=2)

	out, err = (final.output(output, f="mp4", vcodec="libx264", pix_fmt="yuv420p", movflags="+faststart")
		.overwrite_output()
		.run(input=process_tel.stdout.read(), capture_stdout=True, capture_stderr=True)
	)

	print(err.decode())  # Exibir erro detalhado do ffmpeg



	#%% =========================================================== Remove o arquivo auxiliares ===========================================================
	# if deleteaux: 
	# 	try:
	# 		os.remove(temp_file_tel)
	# 	except Exception as e:
	# 		warnings.warn(f"Não foi possível remover o arquivo auxiliar {temp_file_tel} depois do processamento.")