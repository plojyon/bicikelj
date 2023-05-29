# BicikeLJ predictor

Napovedni model za število koles na BicikeLJ postajah.

## Problem
BicikeLJ je sistem za izposojo koles v Ljubljani. Skrbniki skušajo
zagotavljati dostopnost koles s prevažanjem koles med postajami. Da bi
naredili prevažanje bolj učinkovito, želimo napovedati število koles na
postajah vnaprej.

## Podatki
Podatki so na voljo v datoteki `data/bicikelj_train.csv`. Vsebujejo število
koles na vsaki postaji v sklopih (batches) meritev v nekaj minutnih razmakih v
obdobju od 2021-08-02 do 2021-10-01.

Sklopi so med seboj ralično oddaljeni (lahko manjka tudi kak dan), meritve
znotraj sklopa pa so načeloma vsakih ~5 minut.

Na koncu vsakega sklopa želimo napovedati število koles čez 1h in 2h.

## Evalvacija
Za evalvacijsko množico sem vzel zadnji podatek vsakega sklopa, in podatek eno
uro pred koncem sklopa. Iz trenirnih podatkov sem nato zadnji dve uri podatkov
pobrisal, da sem ustvaril okolje, podobno produkcijskemu.

Učinkovitost modela sem ocenjeval s povprečno absolutno napako (MAE).

## Model
Za napovedni model se je najbolje izkazala linearna regresija z Ridge
regularizacijo. Za vsako postajo sem naučil svoj model.

V vsak model damo tabelo števila koles ob vrednostih sledečih značilk:
- število koles na postaji pred 1h
- število koles na postaji pred 2h
- ali se je število koles na postaji povečalo v zadnji uri
- ali se je število koles na postaji povečalo v predzadnji uri
- ali so trenutno šolske počitnice (mesec < september)
- mesec v letu
- ali je vikend
- ali je delovnik
- ali je noč (med 0:00 in 6:00)
- ali je dopoldne (med 6:00 in 12:00)
- ali je popoldne (med 12:00 in 18:00)
- ali je večer (med 18:00 in 24:00)
- ali je temperatura nad 30°C
- ali je temperatura pod 10°C
- ali pada dež
- ali pada več kot 0.3mm dežja
- ali pada več kot 1mm dežja
- ura v dnevu
- dan v letu


## Vremenski podatki
Vremenski podatki so na voljo v datoteki `data/weather.csv`. Vsebujejo
podatke o vremenu, ki jih deli ARSO za samodejno postajo "Ljubljana - Bežigrad".

Ker nekatere vrednosti v stolpcu `rain` manjkajo, sem jih dopolnil z
linearno interpolacijo najbližjih danih vrednosti.

https://meteo.arso.gov.si/met/sl/archive/
