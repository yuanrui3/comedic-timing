"""Ten diagnostic adaptations; semantic review and LM retention are separate."""
from .common import ROOT, atomic_json
import re
import unicodedata

ROWS = [
    ('請神', '一个吝啬鬼请道士请神，问他为什么只请远方的神。道士说，', '附近的神知道你小气都不肯来。', '远方的神正好有空可以请过来。'),
    ('兄弟種田', '兄弟分庄稼，哥哥要上半截。弟弟说好，明年换我拿上半截。哥哥说，', '那我明年就改种芋头。', '那咱明年就交换一下。'),
    ('合伙做酒', '两人合伙酿酒，一个出米，一个出水。出水的说我绝不占你便宜，', '酒好了只要还我那些水。', '酒好了咱们按成本分配。'),
    ('許日子', '仆人说我家主人下辈子才会请客。吝啬的主人听见就骂，', '谁让你随便定下日子的。', '谁让你随便议论主人的。'),
    ('攜燈', '客人喝了一夜的酒，天亮了仆人来接。他怪仆人没带灯，', '再喝一会儿天就又黑了。', '回家路上有段路特别暗。'),
    ('不留飯', '客人坐到中午，主人不留饭。听见鸡叫，主人说这鸡报时不准。客人说，', '我的肚子报时可准得很。', '这只公鸡今天叫得真早。'),
    ('下飯', '父亲让儿子看着墙上的咸鱼下饭。弟弟告状，说哥哥多看了一眼。父亲说，', '那还不得把他给咸死了。', '那就让他少吃一口米饭。'),
    ('蘸酒', '吝啬的父亲只许用筷子蘸酒喝。儿子连蘸两下，父亲责备他说，', '你怎么喝得这么着急啊。', '你怎么不把筷子放下啊。'),
    ('夢美酒', '有人梦见得到一壶好酒，正想温热了喝，却被叫醒。他后悔地说，', '早知道就该喝凉的了。', '早知道就该早点睡了。'),
    ('借水', '客人没分到筷子，便向主人要一碗水。主人问拿来做什么，客人说，', '把手洗净好用手指夹菜。', '刚才走路口渴想喝点水。'),
]

def han(text):
    return re.findall(r'[\u4e00-\u9fff]', text)

def punctuation_signature(text):
    count = 0
    result = []
    for c in text:
        if unicodedata.category(c).startswith('P'):
            result.append((count, c))
        elif not c.isspace():
            count += 1
    return result

def reduced(text):
    return ''.join('。' if c in '。！？.!?…' else '，' if unicodedata.category(c).startswith('P') else c for c in text)

def main():
    records = []
    for i, (title, setup, joke, control) in enumerate(ROWS, 1):
        assert len(han(joke)) == len(han(control)), (title, len(han(joke)), len(han(control)))
        assert punctuation_signature(joke) == punctuation_signature(control), title
        records.append({
            'id': f'zh{i:03}', 'language': 'zh', 'source_title': title,
            'source_url': 'https://zh.wikisource.org/w/index.php?title=笑林廣記/貪吝部&oldid=2472153#'+title,
            'source_license': 'Public domain historical text; Wikisource PD-old notice',
            'source_author': '遊戲主人 (traditional attribution), 笑林廣記',
            'adaptation': 'New AI-authored simplified-Chinese retelling; not a verbatim historical quotation. Control newly generated for this study.',
            'annotation_status': 'provisional_agent_split_human_review_deferred',
            'manipulation_status': 'unscored', 'setup': setup, 'joke': joke, 'control': control,
            'ending_han_count': len(han(joke)),
            'reduced': {'setup': reduced(setup), 'joke': reduced(joke), 'control': reduced(control)},
        })
    atomic_json(ROOT/'data/scripts/open_zh_diagnostic.json', records)
    print(f'{len(records)} pairs: ending lengths and punctuation positions match.')

if __name__ == '__main__':
    main()
