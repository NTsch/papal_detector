xquery version "3.1";
declare namespace atom = "http://www.w3.org/2005/Atom";
declare namespace cei = "http://www.monasterium.net/NS/cei";
declare namespace xrx = "http://www.monasterium.net/NS/xrx";
declare namespace eag = "http://www.archivgut-online.de/eag";
declare namespace tei = "http://www.tei-c.org/ns/1.0";
declare namespace ead="urn:isbn:1-931666-22-9";

(:let $bestaende-output := doc('bestaende_output.xml')
let $papst-at-start := doc('correspondence_list/papst_at_start_filtered_img.xml')
let $kloster-imgs := doc('correspondence_list/klosterbestaende_bilder.xml')

let $classified-as-papal := $bestaende-output//charter[prediction='papal' or prediction='papal_simple'][xs:float(confidence/text()) > 0.85]/image/tokenize(text(), '/')[last()]
let $kloster-img-ids := 
    for $img in $kloster-imgs/tei:results/tei:charter
    where some $url in $classified-as-papal satisfies contains($img/@img/data(), $url)
    return $img/@id/data()
let $matches :=
    for $id in $kloster-img-ids
    where not(
      some $charter in $papst-at-start/tei:charters/cei:charter
      satisfies $charter/@atom:id/data() = $id
    )
    return $id
for $match in $matches return <result>{$match}</result>:)

<charters>{
let $ids := doc('/db/niklas/papsturkunden/discovered_charters.xml')//result/text()
for $charter in collection('/db/mom-data/metadata.charter.public')/atom:entry[atom:id/text() = $ids]
return <charter atom:id="{$charter/atom:id/text()}">{$charter//atom:content/cei:text/cei:body/cei:chDesc/cei:abstract}</charter>
}</charters>
