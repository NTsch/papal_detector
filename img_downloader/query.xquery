xquery version "3.1";
declare namespace atom = "http://www.w3.org/2005/Atom";
declare namespace cei = "http://www.monasterium.net/NS/cei";
declare namespace xrx = "http://www.monasterium.net/NS/xrx";
declare namespace eag = "http://www.archivgut-online.de/eag";
declare namespace rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#";
declare namespace skos="http://www.w3.org/2004/02/skos/core#";
declare default element namespace 'http://www.tei-c.org/ns/1.0';

for $charter in doc('klosterbestaende_bilder.xml')//charter
return concat($charter/@img/data(), '&#xA;')