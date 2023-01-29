import getTweet
import detectDepression
import generateLove
import sendMedicine


# Algorithm for detecting depression and presenting virtual hug memes or kind words
# maybe search in #dark or #depressed
# tweet = getTweet()
# depressedHuman = detectDepression(tweet)
# if depressedHuman:
#   love = generateLove()
#   sendMedicine(love)

tweet = getTweet.getTweet()
depressedHuman = detectDepression(tweet)
if depressedHuman:
    love = generateLove()
    
    