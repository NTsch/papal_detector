xquery version "3.1";
declare namespace atom = "http://www.w3.org/2005/Atom";
declare namespace cei = "http://www.monasterium.net/NS/cei";
declare namespace xrx = "http://www.monasterium.net/NS/xrx";
declare namespace eag = "http://www.archivgut-online.de/eag";
declare namespace rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#";
declare namespace skos="http://www.w3.org/2004/02/skos/core#";
declare default element namespace 'http://www.tei-c.org/ns/1.0';

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

for $charter in doc('klosterbestaende_bilder.xml')//charter
return $charter/@img/data()