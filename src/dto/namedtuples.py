from collections import namedtuple

Datum = namedtuple('Datum', 'id ref cite offsets author is_test facet')
Offsets = namedtuple('Offsets', 'marker cite ref')
Article = namedtuple('Article', 'content')


