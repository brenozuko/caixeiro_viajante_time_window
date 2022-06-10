# P2 de Programação Linear - Fatec Ribeirão Preto

Alunos envolvidos:

Breno Zukowski
Henrique Ribeiro
Jean Silva
Paola Capita


## Sobre o trabalho:

Neste trabalho modelamos o problema do caixeiro viajante com deadlines, visando minizar o atraso como descrito no enunciado: 

> Considere um veículo que deve partir de um ponto inicial, visitar n localidades e retornar ao
ponto de partida após as visitas. Considere adicionalmente que cada localidade a ser visitada
tem um prazo limite para receber a visita. Atrasos são permitidos, porém uma multa que
aumenta com o tempo de atraso é imposta. Os dados apresentam as coordenadas cartesianas
das localidades. Utilize a distância euclidiana (em linha reta) entre as localidades como tempo
de percurso entre elas. Formule modelo de otimização que determine a rota que minimize o
atraso total nas visitas.

## Como navegar neste repositório:

A implementação do modelo se encontra na pasta `notebooks` em sua verão `.py` e `.ipynb`

O artigo desenvolvido em LATEX na pasta `artigo`

Os exports com os resultados obtidos no processamento das instâncias na pasta `dados`

Além de algumas referências utilizadas na pasta `referencias`


## Artigo:

**Resumo:**

O problema do caixeiro viajante (PCV) é um clássico da literatura matemática, onde um vendedor deve visitar um conjunto de N cidades 
uma única vez e retornar ao seu ponto de origem, através de uma rota que minimiza a distância percorrida. Este trabalho propõe uma solução 
via modelo de programação matemática, para uma variante do problema original que conta com tempos limite para chegada a cada cidade, 
em que atrasos são permitidos e têm-se como objetivo minimizar o atraso geral da rota. Com a utilização da biblioteca PyMathProg exploramos a 
complexidade do PCV e os resultados estão expostos neste artigo.


*O artigo resultado deste projeto pode ser encontrado em:*

https://pt.slideshare.net/BrenoZukowski2/caixeiro-viajante-com-deadline-aplicado-para-a-minimizac-ao-de-atraso-de-chegada-em-n-localidades
