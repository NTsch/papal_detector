xquery version "3.1";
declare namespace atom = "http://www.w3.org/2005/Atom";
declare namespace cei = "http://www.monasterium.net/NS/cei";
declare namespace xrx = "http://www.monasterium.net/NS/xrx";
declare namespace eag = "http://www.archivgut-online.de/eag";
declare namespace rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#";
declare namespace skos="http://www.w3.org/2004/02/skos/core#";
declare namespace tei="http://www.tei-c.org/ns/1.0";
(:declare default element namespace 'http://www.tei-c.org/ns/1.0';:)

(:<results>{
for $fond in doc('klosterbestaende.xml')//result/text()
let $img-base-url := collection(concat('/db/mom-data/metadata.fond.public/', $fond))/xrx:preferences/xrx:param[@name="image-server-base-url"]/text()
for $collection in concat('/db/mom-data/metadata.charter.public/', $fond)
for $charter in collection($collection)/atom:entry
let $imgs := $charter/atom:content/cei:text/cei:body/cei:chDesc/cei:witnessOrig/cei:figure[not(cei:graphic/@n='thumbnail')][contains(cei:graphic/@url, 'r.') or (position() = 1 and not(contains(cei:graphic/@url, 'v.')))]
where $imgs
return 
    <charter id="{$charter/atom:id/text()}" img="{
    let $recto := $imgs[1]/cei:graphic
    return if (contains($recto/@url, 'http')) then <img>{$recto/@url/data()}</img>
    else <img>{concat($img-base-url, '/', $recto/@url/data())}</img>
    }"/>
}</results>:)

(:for $charter in doc('klosterbestaende_bilder.xml')//tei:charter
return concat($charter/@img/data(), '
'):)

(:how many charters were predicted per fond?:)

(:<fonds>{
let $papal_output := doc('bestaende_output.xml')
let $bestaende_bilder := doc('klosterbestaende_bilder.xml')

let $fonds :=
for $charter in $papal_output//charter[prediction[not(contains(text(), 'non'))]]
    let $img := substring-after($charter/image/text(), 'klosterbestaende_writable_area_input/')
    let $corresp-char := $bestaende_bilder//tei:charter[ends-with(lower-case(@img), lower-case($img))][1]
    let $fond-tokens := tokenize(substring-after($corresp-char/@id/data(), 'charter/'), '/')
    let $fond := concat($fond-tokens[1], '/', $fond-tokens[2])
    where not($fond = '/')
    return $fond
    
for $f in distinct-values($fonds)
let $count := count($fonds[. = $f])
order by $count descending
return <fond name="{$f}" count="{$count}"/>
}</fonds>:)

(: Schnittmenge papal_text & papal_cv :)

(:<fonds>{
let $papal_output := doc('bestaende_output.xml')
let $bestaende_bilder := doc('klosterbestaende_bilder.xml')
let $pas_ids := doc('papst_at_start_filtered_img.xml')//cei:charter/@atom:id

let $fonds :=
  for $charter in $papal_output//charter[prediction[not(contains(., 'non'))]]
  let $img := substring-after($charter/image, 'klosterbestaende_writable_area_input/')
  let $corresp := $bestaende_bilder//tei:charter[@id = $pas_ids and ends-with(lower-case(@img), lower-case($img))][1]
  let $tokens := tokenize(substring-after($corresp/@id, 'charter/'), '/')
  let $fond := string-join($tokens[position() le 2], '/')
  where $fond
  return $fond

for $f in distinct-values($fonds)
let $count := count($fonds[. = $f])
order by $count descending
return <fond name="{$f}" count="{$count}"/>
}</fonds>:)