"""
Extract tags from a text.
Inspired from:
https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/llm-structured/instructor_generate.py
https://github.com/instructor-ai/instructor/blob/main/docs/concepts/retrying.md
"""

from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List

import instructor
from instructor.exceptions import InstructorRetryException


class Section(BaseModel):
    tag: str = Field(
        description="Human activity mentioned in this section of the text, \
        such as politics, sports, fishing, agriculture, tourism..."
    )
    keyword: str = Field(
        description="Word from the text justifying the tag. Example: the keyword 'football' justifies the tag 'sports'"
    )
    excerpt: str = Field(
        description="Small excerpt from the text giving more context to the extracted word. Example for the word football: 'la coupe du monde de football'"
    )


class MetadataExtraction(BaseModel):
    """Extracted metadata about an example from the Modal examples repo."""

    summary: str = Field(
        ..., description="A brief summary of the text (less than 30 words)."
    )
    location: str = Field(
        ..., description="The place where human activities take place, if any."
    )
    sections: List[Section] = Field(
        description="A list of small excerpts of the document mentioning human activities."
    )


client = instructor.from_openai(
    OpenAI(
        base_url="http://localhost:5005/v1",
        api_key="ollama",  # required, but unused
    ),
    mode=instructor.Mode.JSON,
)

queries = [
    "Bonjour, Connaissez vous l'Agave Americana? Communément appelé Agave américain ou Agave d'Amérique. Il est également appelé choka bleu à La Réunion. Il s'agit d'une espèce originaire d'Amérique du Nord. Elle est considérée comme une espèce invasive dans le sud de la France, notamment dans le Parc national des Calanques où des campagnes d'arrachage ont régulièrement lieu . 📷 Corniche de L'esterel - 13 Septembre 2017",
    "La pêche fantôme, une hécatombe silencieuse mais bien réelle. Le problème est connu depuis les années 1960, quand les flottes de pêche ont commencé à troquer leurs filets en fibre naturelle pour le plastique. Plus efficaces et plus maniables, les engins de pêche (casiers, sennes, chaluts, filets) ont aussi vu leur espérance de vie en mer s'accroître considérablement. De ce fait, qu'il soit perdu ou volontairement abandonné, un filet de pêche en nylon qui peut atteindre 600 m de long, reste pêchant pendant des mois, voire des années, piégeant tortues, phoques, baleines, dauphins ou autres oiseaux de mer, souvent condamnés à une lente, douloureuse et imperceptible agonie. C’est un cercle vicieux complètement évitable : des poissons se retrouvent pris dans les mailles, en attirent d'autres, souvent plus gros, qui s’y coincent à leur tour et meurent d'asphyxie au bout de 24 à 48 heures. Le filet est aujourd’hui le macro-déchet le plus polluant des océans dont la pêche, qu’elle soit industrielle ou artisanale, supporte toute la responsabilité. Si ce phénomène mondial reste difficile à quantifier, une enquête de sciences participatives (Fish & Clik) a cependant permis de recenser 27 000 engins ou débris d'engins de pêche sur le littoral français en seulement deux ans, entre la Bretagne et les Hauts-de-France. Chaque année, près de 80 000 kilomètres carrés de filets, l'équivalent de la superficie de l'Écosse, dérivent et disparaissent dans les fonds marins, emportant gratuitement avec eux des millions de victimes dont l’humanité ne tire aucun profit. Il ne peut pas y avoir de pêche durable dans un monde dominé par l’économie, pour une espèce qui prolifère avec toujours plus d’appétit, quoi que promettent les labels et les lobbies. https://www.rtbf.be/article/les-filets-fantomes-fleau-invisible-des-oceans-11205593 Baleine à bosse prise dans un filet de pêche au large des baléares en 2022, miraculeusement sauvée par des plongeurs chevronnés.",
    "📢ALERTE 🚨 ATTENTION À partir du vendredi 16 février, des vagues de hauts vents du nord-ouest et du sud-ouest de 25 à 45 km/h et des vagues de 2 à 3 mètres de haut sont prévues selon des informations partagées par la Protection Civile. Cette condition entraînera également une réduction de la zone de plage, des marées récurrentes et des courants de chaluts à l'intérieur de la Bahia, dans les zones de mer libre, Pie de la Cuesta, Puerto Marques, La Roquettea, Bonfil et Pie de la Cuesta. Nous vous invitons à prendre en compte les recommandations suivantes lors de votre visite en mer. 🌊",
    "🔴 2021 - Accord de Pêche UE-Gabon : préoccupant pour les écosystèmes marins et les communautés côtières Caroline Roose ( Euro-deputée) Nous votons aujourd'hui en séance plénière le nouvel accord de pêche entre l’UE et le Gabon, qui est :- préoccupant pour les écosystèmes marins et les communautés côtières- flou sur la manière dont l’argent public européen sera utilisé Après avoir déposé un amendement de rejet de l'accord, et après avoir demandé des précisions sur l’impact des chaluts de fonds, la transparence, et la façon dont les fonds versés aideront concrètement la pêche artisanale et augmenteront les retombées socio-économiques, je voterai contre cet accord. 🔴 Accord de pêche UE-Gabon : préoccupant pour les écosystèmes marins et les communautés côtières Le 27 octobre les député·e·s de la commission de la pêche du Parlement européen ont approuvé le renouvellement de l’accord de pêche entre l’Union européenne et le Gabon. Le nouveau protocole fixe les conditions d’accès à 33 navires européens, principalement des thoniers senneurs français et espagnols, qui pêcheront le thon dans les eaux gabonaises pour les 5 prochaines années, en l’échange d’une contribution financière totale de l’UE de 13 millions d’euros. La majeure partie de cette somme correspond à une compensation financière pour l’accès aux eaux et aux ressources halieutiques du Gabon tandis que le reste est alloué au soutien au secteur de la pêche au Gabon (contrôle des pêches, durabilité, soutien à la pêche artisanale, etc.). Le protocole prévoit également de donner l’accès à 4 chalutiers ciblant les crustacés d’eau profonde, dans le cadre d’une pêche exploratoire. Pour Caroline Roose (Verts/ALE), cet accord de pêche est préoccupant : « Cet accord constitue une menace pour les populations de poissons et les écosystèmes marins. Bien que la plupart des populations de poissons sont surexploitées ou non évaluées dans la région du Gabon, l’accord permet à 4 chalutiers de fonds de mener des « pêches exploratoires ». Les études scientifiques sont pourtant très claires : ces engins de pêche ont des impacts dévastateurs sur les fonds marins et les captures accidentelles d’espèces non ciblées1. Pour preuve, les annexes du protocole indiquent des limites de prises accessoires autorisées élevées. Cet accord ne profite pas aux populations locales. Du fait du manque d’infrastructures pour le débarquement et les activités de transformation du poisson, les thons pêchés ne seront pas débarqués au Gabon. La valeur ajoutée pour les gabonais est donc très faible et l’accord profite surtout aux industriels européens2. Cet accord reste flou sur la manière dont l’argent public européen sera utilisé. L’évaluation du précédent protocole montre clairement que le soutien sectoriel versé par l’UE n’a pas été utilisé de façon optimale. Dans un pays comme le Gabon, où les droits humains ont été bafoués ces dernières années (voir la résolution du Parlement européen en 2017), et à la lumière de l’affaire récente des Pandora Papers dans laquelle le nom du président du pays a été cité, nous avons besoin de garanties de transparence sur la façon dont l’argent sera utilisé une fois dans les mains du gouvernement gabonais. L’Union européenne doit mettre ses accords de pêche internationaux en ligne avec ses objectifs environnementaux et de développement. Nous devons cesser de surexploiter les ressources marines des pays en développement alors que nous voulons être les champions de la biodiversité.» Les eurodéputé·e·s écologistes, qui avaient déposé un amendement de rejet de l’accord, ont introduit une question écrite à la Commission européenne avec d’autres élu·e·s pour demander des précisions sur l’impact des chaluts de fonds, la transparence, et la façon dont les fonds versés aideront concrètement la pêche artisanale et augmenteront les retombées socio-économiques. La prochaine étape sera le vote final de l’accord en séance plénière. [1] Une étude sur les pêcheries gabonaises indique que lors de campagnes océanographiques de pêche à la crevette d’eau profonde avec des engins de pêche expérimentaux, la composition des captures a montré des niveaux de prises accessoires importants. Voir Landry Ekouala. Le développement durable et le secteur des pêches et de l’aquaculture au Gabon : une étude de la gestion durable des ressources halieutiques et leur écosystème dans les provinces de l’Estuaire et de l’Ogooué Maritime. Histoire. Université du Littoral Côte d’Opale, 2013. Français. [2] L’évaluation ex-post du protocole précédent (2013-2016) souligne la faible valeur ajoutée totale reçue par le Gabon (11%), en raison de l’absence d’infrastructures de débarquement et de transformation du thon au Gabon. Elle mentionne également des retards et des incohérences dans la transmission des données par les États membres. De plus, au vu du manque d’infrastructures de formation, les marins embarqués sur les bateaux européens ne seront probablement pas gabonais.",
]


for query in queries:
    print(query)
    try:
        resp = client.chat.completions.create(
            model="deepseek-r1:70b",
            messages=[
                {
                    "role": "user",
                    "content": f"Extract the metadata for this text. \n\n-----TEXT BEGINS-----{query}-----TEXT ENDS-----\n\n",
                }
            ],
            response_model=MetadataExtraction,
        )
        print(resp.model_dump_json(indent=2))

    except InstructorRetryException:
        print("InstructorRetryException")
